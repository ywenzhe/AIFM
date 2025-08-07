extern "C" {
#include <runtime/runtime.h>
}
#include "thread.h"

#include "array.hpp"
#include "deref_scope.hpp"
#include "device.hpp"
#include "helpers.hpp"
#include "manager.hpp"
#include "snappy.h"
#include "stats.hpp"
#include "zipf.hpp"

// crypto++
#include "cryptopp/aes.h"
#include "cryptopp/filters.h"
#include "cryptopp/modes.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <fstream>

using namespace far_memory;

#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))

namespace far_memory {
class WebServiceBenchmark {
private:
  // FarMemManager配置 - 支持可配置的缓存大小
  uint64_t kCacheSize;
  constexpr static uint64_t kFarMemSize = (20ULL << 30); // 20 GB
  constexpr static uint32_t kNumGCThreads = 12;
  constexpr static uint32_t kNumConnections = 300;

  // Hashtable配置 - 10GB数据集
  constexpr static uint32_t kKeyLen = 12;
  constexpr static uint32_t kValueLen = 4;
  constexpr static uint32_t kLocalHashTableNumEntriesShift = 28; // 可调整
  constexpr static uint32_t kRemoteHashTableNumEntriesShift = 30; // 10GB数据集
  constexpr static uint64_t kRemoteHashTableSlabSize = (10ULL << 30) * 1.05; // 10GB + 5%
  constexpr static uint32_t kNumKVPairs = 1 << 28; // ~268M pairs for 10GB dataset

  // Array配置 - 16GB数据集
  constexpr static uint32_t kNumArrayEntries = (16ULL << 30) / 8192; // 16GB / 8K = ~2M entries
  constexpr static uint32_t kArrayEntrySize = 8192; // 8K

  // Runtime配置
  constexpr static uint32_t kNumMutatorThreads = 40;
  constexpr static double kZipfParamS = 0.85;
  constexpr static uint32_t kNumKeysPerRequest = 32;
  constexpr static uint32_t kNumReqs = kNumKVPairs / kNumKeysPerRequest;
  constexpr static uint32_t kLog10NumKeysPerRequest =
      helpers::static_log(10, kNumKeysPerRequest);
  constexpr static uint32_t kReqLen = kKeyLen - kLog10NumKeysPerRequest;
  constexpr static uint32_t kReqSeqLen = kNumReqs;

  // 基准测试配置
  constexpr static uint32_t kPrintPerIters = 8192;
  constexpr static uint32_t kMaxPrintIntervalUs = 1000 * 1000; // 1秒
  constexpr static uint32_t kBenchmarkDurationSec = 60; // 基准测试持续时间

  struct Req {
    char data[kReqLen];
  };

  struct Key {
    char data[kKeyLen];
  };

  union Value {
    uint32_t num;
    char data[kValueLen];
  };

  struct ArrayEntry {
    uint8_t data[kArrayEntrySize];
  };

  struct alignas(64) Cnt {
    uint64_t c;
  };

  using AppArray = Array<ArrayEntry, kNumArrayEntries>;

  std::unique_ptr<std::mt19937> generators[helpers::kNumCPUs];
  std::unique_ptr<Req[]> all_gen_reqs;
  uint32_t all_zipf_req_indices[helpers::kNumCPUs][kReqSeqLen];
  
  Cnt req_cnts[kNumMutatorThreads];
  Cnt local_array_miss_cnts[kNumMutatorThreads];
  Cnt local_hashtable_miss_cnts[kNumMutatorThreads];
  Cnt per_core_req_idx[helpers::kNumCPUs];

  std::atomic_flag flag;
  std::atomic<bool> benchmark_running{true};
  uint64_t benchmark_start_us = 0;
  uint64_t prev_us = 0;
  std::vector<double> mops_records;
  std::vector<double> hashtable_miss_rate_records;
  std::vector<double> array_miss_rate_records;
  std::vector<uint64_t> latency_samples;

  // 性能统计
  struct BenchmarkResults {
    double avg_mops;
    double avg_hashtable_miss_rate;
    double avg_array_miss_rate;
    double total_runtime_sec;
    uint64_t total_requests;
    uint64_t cache_size_mb;
    double p99_latency_us;
    double avg_latency_us;
  };

  unsigned char key[CryptoPP::AES::DEFAULT_KEYLENGTH];
  unsigned char iv[CryptoPP::AES::BLOCKSIZE];
  std::unique_ptr<CryptoPP::CBC_Mode_ExternalCipher::Encryption> cbcEncryption;
  std::unique_ptr<CryptoPP::AES::Encryption> aesEncryption;

  inline void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len,
                                          char *array) {
    uint32_t len = 0;
    while (n) {
      auto digit = n % 10;
      array[len++] = digit + '0';
      n = n / 10;
    }
    while (len < suffix_len) {
      array[len++] = '0';
    }
    std::reverse(array, array + suffix_len);
  }

  inline void random_string(char *data, uint32_t len) {
    BUG_ON(len <= 0);
    preempt_disable();
    auto guard = helpers::finally([&]() { preempt_enable(); });
    auto &generator = *generators[get_core_num()];
    std::uniform_int_distribution<int> distribution('a', 'z' + 1);
    for (uint32_t i = 0; i < len; i++) {
      data[i] = char(distribution(generator));
    }
  }

  inline void random_req(char *data, uint32_t tid) {
    auto tid_len = helpers::static_log(10, kNumMutatorThreads);
    random_string(data, kReqLen - tid_len);
    append_uint32_to_char_array(tid, tid_len, data + kReqLen - tid_len);
  }

  inline uint32_t random_uint32() {
    preempt_disable();
    auto guard = helpers::finally([&]() { preempt_enable(); });
    auto &generator = *generators[get_core_num()];
    std::uniform_int_distribution<uint32_t> distribution(
        0, std::numeric_limits<uint32_t>::max());
    return distribution(generator);
  }

  void prepare(GenericConcurrentHopscotch *hopscotch) {
    std::cout << "准备Hashtable数据集 (10GB)..." << std::endl;
    
    for (uint32_t i = 0; i < helpers::kNumCPUs; i++) {
      std::random_device rd;
      generators[i].reset(new std::mt19937(rd()));
    }
    
    memset(key, 0x00, CryptoPP::AES::DEFAULT_KEYLENGTH);
    memset(iv, 0x00, CryptoPP::AES::BLOCKSIZE);
    aesEncryption.reset(
        new CryptoPP::AES::Encryption(key, CryptoPP::AES::DEFAULT_KEYLENGTH));
    cbcEncryption.reset(
        new CryptoPP::CBC_Mode_ExternalCipher::Encryption(*aesEncryption, iv));
    
    std::vector<rt::Thread> threads;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(rt::Thread([&, tid]() {
        auto num_reqs_per_thread = kNumReqs / kNumMutatorThreads;
        auto req_offset = tid * num_reqs_per_thread;
        auto *thread_gen_reqs = &all_gen_reqs.get()[req_offset];
        
        for (uint32_t i = 0; i < num_reqs_per_thread; i++) {
          Req req;
          random_req(req.data, tid);
          Key key;
          memcpy(key.data, req.data, kReqLen);
          
          for (uint32_t j = 0; j < kNumKeysPerRequest; j++) {
            append_uint32_to_char_array(j, kLog10NumKeysPerRequest,
                                        key.data + kReqLen);
            Value value;
            value.num = (j ? 0 : req_offset + i);
            DerefScope scope;
            hopscotch->put(scope, kKeyLen, (const uint8_t *)key.data, kValueLen,
                           (uint8_t *)value.data);
          }
          thread_gen_reqs[i] = req;
          
          // 进度报告
          if (i % 10000 == 0) {
            std::cout << "线程 " << tid << " 已处理 " << i << "/" << num_reqs_per_thread << " 请求" << std::endl;
          }
        }
      }));
    }
    
    for (auto &thread : threads) {
      thread.Join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto prep_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Hashtable数据准备完成，耗时: " << prep_duration << "秒" << std::endl;
    
    // 生成Zipf分布的请求索引
    preempt_disable();
    zipf_table_distribution<> zipf(kNumReqs, kZipfParamS);
    auto &generator = generators[get_core_num()];
    constexpr uint32_t kPerCoreWinInterval = kReqSeqLen / helpers::kNumCPUs;
    for (uint32_t i = 0; i < kReqSeqLen; i++) {
      auto rand_idx = zipf(*generator);
      for (uint32_t j = 0; j < helpers::kNumCPUs; j++) {
        all_zipf_req_indices[j][(i + (j * kPerCoreWinInterval)) % kReqSeqLen] =
            rand_idx;
      }
    }
    preempt_enable();
  }

  void prepare(AppArray *array) {
    std::cout << "准备Array数据集 (16GB)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 初始化Array数据
    std::vector<rt::Thread> threads;
    constexpr uint32_t kEntriesPerThread = kNumArrayEntries / kNumMutatorThreads;
    
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(rt::Thread([&, tid]() {
        uint32_t start_idx = tid * kEntriesPerThread;
        uint32_t end_idx = (tid == kNumMutatorThreads - 1) ? kNumArrayEntries : (tid + 1) * kEntriesPerThread;
        
        for (uint32_t i = start_idx; i < end_idx; i++) {
          DerefScope scope;
          auto &entry = array->at</* NT = */ false>(scope, i);
          // 填充随机数据
          for (uint32_t j = 0; j < kArrayEntrySize; j++) {
            entry.data[j] = (uint8_t)(i + j);
          }
          
          if ((i - start_idx) % 10000 == 0) {
            std::cout << "线程 " << tid << " 已初始化 " << (i - start_idx) << "/" << (end_idx - start_idx) << " 数组条目" << std::endl;
          }
        }
      }));
    }
    
    for (auto &thread : threads) {
      thread.Join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto prep_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Array数据准备完成，耗时: " << prep_duration << "秒" << std::endl;
  }

  void consume_array_entry(const ArrayEntry &entry) {
    std::string ciphertext;
    CryptoPP::StreamTransformationFilter stfEncryptor(
        *cbcEncryption, new CryptoPP::StringSink(ciphertext));
    stfEncryptor.Put((const unsigned char *)&entry.data, sizeof(entry));
    stfEncryptor.MessageEnd();
    std::string compressed;
    snappy::Compress(ciphertext.c_str(), ciphertext.size(), &compressed);
    auto compressed_len = compressed.size();
    ACCESS_ONCE(compressed_len);
  }

  void collect_performance_stats() {
    if (!flag.test_and_set()) {
      preempt_disable();
      auto us = microtime();
      uint64_t sum_reqs = 0;
      uint64_t sum_hashtable_misses = 0;
      uint64_t sum_array_misses = 0;
      
      for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
        sum_reqs += ACCESS_ONCE(req_cnts[i].c);
        sum_hashtable_misses += ACCESS_ONCE(local_hashtable_miss_cnts[i].c);
        sum_array_misses += ACCESS_ONCE(local_array_miss_cnts[i].c);
      }
      
      if (us - prev_us > kMaxPrintIntervalUs) {
        auto mops = ((double)(sum_reqs) / (us - benchmark_start_us)) * 1000000.0 / 1000000.0;
        auto hashtable_miss_rate = (double)sum_hashtable_misses / (kNumKeysPerRequest * sum_reqs);
        auto array_miss_rate = (double)sum_array_misses / sum_reqs;
        
        mops_records.push_back(mops);
        hashtable_miss_rate_records.push_back(hashtable_miss_rate);
        array_miss_rate_records.push_back(array_miss_rate);
        
        std::cout << "当前性能 - MOPS: " << mops 
                  << ", Hashtable缺失率: " << hashtable_miss_rate 
                  << ", Array缺失率: " << array_miss_rate << std::endl;
        
        prev_us = us;
        
        // 检查是否达到基准测试时间
        if ((us - benchmark_start_us) / 1000000 >= kBenchmarkDurationSec) {
          benchmark_running.store(false);
        }
      }
      preempt_enable();
      flag.clear();
    }
  }

  void bench(GenericConcurrentHopscotch *hopscotch, AppArray *array) {
    std::cout << "开始基准测试 (持续" << kBenchmarkDurationSec << "秒)..." << std::endl;
    
    std::vector<rt::Thread> threads;
    benchmark_start_us = microtime();
    prev_us = benchmark_start_us;
    
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(rt::Thread([&, tid]() {
        uint32_t cnt = 0;
        std::vector<uint64_t> local_latencies;
        local_latencies.reserve(1000);
        
        while (benchmark_running.load()) {
          if (unlikely(cnt++ % kPrintPerIters == 0)) {
            preempt_disable();
            collect_performance_stats();
            preempt_enable();
          }
          
          auto request_start = microtime();
          
          preempt_disable();
          auto core_num = get_core_num();
          auto req_idx = all_zipf_req_indices[core_num][per_core_req_idx[core_num].c];
          if (unlikely(++per_core_req_idx[core_num].c == kReqSeqLen)) {
            per_core_req_idx[core_num].c = 0;
          }
          preempt_enable();

          auto &req = all_gen_reqs.get()[req_idx];
          Key key;
          memcpy(key.data, req.data, kReqLen);
          uint32_t array_index = 0;
          
          // Hashtable操作
          {
            DerefScope scope;
            for (uint32_t i = 0; i < kNumKeysPerRequest; i++) {
              append_uint32_to_char_array(i, kLog10NumKeysPerRequest,
                                          key.data + kReqLen);
              Value value;
              uint16_t value_len;
              bool forwarded = false;
              hopscotch->_get(kKeyLen, (const uint8_t *)key.data,
                              &value_len, (uint8_t *)value.data, &forwarded);
              ACCESS_ONCE(local_hashtable_miss_cnts[tid].c) += forwarded;
              array_index += value.num;
            }
          }
          
          // Array操作
          {
            array_index %= kNumArrayEntries;
            DerefScope scope;
            ACCESS_ONCE(local_array_miss_cnts[tid].c) +=
                !array->ptrs_[array_index].meta().is_present();
            const auto &array_entry = array->at</* NT = */ true>(scope, array_index);
            preempt_disable();
            consume_array_entry(array_entry);
            preempt_enable();
          }
          
          auto request_end = microtime();
          local_latencies.push_back(request_end - request_start);
          
          ACCESS_ONCE(req_cnts[tid].c)++;
        }
        
        // 收集延迟样本
        if (!local_latencies.empty()) {
          std::lock_guard<std::mutex> lock(latency_mutex);
          latency_samples.insert(latency_samples.end(), 
                                local_latencies.begin(), local_latencies.end());
        }
      }));
    }
    
    for (auto &thread : threads) {
      thread.Join();
    }
  }

  BenchmarkResults calculate_results() {
    BenchmarkResults results = {};
    
    // 计算平均性能指标
    if (!mops_records.empty()) {
      results.avg_mops = std::accumulate(mops_records.begin(), mops_records.end(), 0.0) / mops_records.size();
      results.avg_hashtable_miss_rate = std::accumulate(hashtable_miss_rate_records.begin(), hashtable_miss_rate_records.end(), 0.0) / hashtable_miss_rate_records.size();
      results.avg_array_miss_rate = std::accumulate(array_miss_rate_records.begin(), array_miss_rate_records.end(), 0.0) / array_miss_rate_records.size();
    }
    
    // 计算总请求数
    for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
      results.total_requests += req_cnts[i].c;
    }
    
    results.total_runtime_sec = (microtime() - benchmark_start_us) / 1000000.0;
    results.cache_size_mb = kCacheSize / (1024 * 1024);
    
    // 计算延迟统计
    if (!latency_samples.empty()) {
      std::sort(latency_samples.begin(), latency_samples.end());
      results.avg_latency_us = std::accumulate(latency_samples.begin(), latency_samples.end(), 0.0) / latency_samples.size();
      results.p99_latency_us = latency_samples[latency_samples.size() * 0.99];
    }
    
    return results;
  }

  void save_results(const BenchmarkResults &results, const std::string &filename) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
      file << results.cache_size_mb << ","
           << results.avg_mops << ","
           << results.avg_hashtable_miss_rate << ","
           << results.avg_array_miss_rate << ","
           << results.total_runtime_sec << ","
           << results.total_requests << ","
           << results.avg_latency_us << ","
           << results.p99_latency_us << std::endl;
      file.close();
    }
  }

  std::mutex latency_mutex;

public:
  WebServiceBenchmark(uint64_t cache_size) : kCacheSize(cache_size * Region::kSize) {
    // 动态分配大数组避免栈溢出
    all_gen_reqs = std::make_unique<Req[]>(kNumReqs);
  }

  void do_work(FarMemManager *manager) {
    auto hopscotch = std::unique_ptr<GenericConcurrentHopscotch>(
        manager->allocate_concurrent_hopscotch_heap(
            kLocalHashTableNumEntriesShift, kRemoteHashTableNumEntriesShift,
            kRemoteHashTableSlabSize));
    
    prepare(hopscotch.get());
    
    auto array_ptr = std::unique_ptr<AppArray>(
        manager->allocate_array_heap<ArrayEntry, kNumArrayEntries>());
    array_ptr->disable_prefetch();
    prepare(array_ptr.get());
    
    bench(hopscotch.get(), array_ptr.get());
    
    // 计算并保存结果
    auto results = calculate_results();
    
    std::cout << "\n=== 基准测试结果 ===" << std::endl;
    std::cout << "缓存大小: " << results.cache_size_mb << " MB" << std::endl;
    std::cout << "平均MOPS: " << results.avg_mops << std::endl;
    std::cout << "Hashtable缺失率: " << results.avg_hashtable_miss_rate << std::endl;
    std::cout << "Array缺失率: " << results.avg_array_miss_rate << std::endl;
    std::cout << "总运行时间: " << results.total_runtime_sec << " 秒" << std::endl;
    std::cout << "总请求数: " << results.total_requests << std::endl;
    std::cout << "平均延迟: " << results.avg_latency_us << " μs" << std::endl;
    std::cout << "P99延迟: " << results.p99_latency_us << " μs" << std::endl;
    
    save_results(results, "benchmark_results.csv");
  }

  void run(netaddr raddr) {
    BUG_ON(madvise(all_gen_reqs.get(), sizeof(Req) * kNumReqs, MADV_HUGEPAGE) != 0);
    std::unique_ptr<FarMemManager> manager =
        std::unique_ptr<FarMemManager>(FarMemManagerFactory::build(
            kCacheSize, kNumGCThreads,
            new TCPDevice(raddr, kNumConnections, kFarMemSize)));
    do_work(manager.get());
  }
};
} // namespace far_memory

int argc;
std::unique_ptr<WebServiceBenchmark> test;

void _main(void *arg) {
  char **argv = (char **)arg;
  std::string ip_addr_port(argv[1]);
  uint64_t cache_size_regions = std::stoull(argv[2]);
  
  auto raddr = helpers::str_to_netaddr(ip_addr_port);
  test = std::make_unique<WebServiceBenchmark>(cache_size_regions);
  test->run(raddr);
}

int main(int _argc, char *argv[]) {
  int ret;

  if (_argc < 4) {
    std::cerr << "用法: [cfg_file] [ip_addr:port] [cache_size_regions]" << std::endl;
    std::cerr << "示例: ./benchmark_main client.config 192.168.1.100:9999 563" << std::endl;
    return -EINVAL;
  }

  char conf_path[strlen(argv[1]) + 1];
  strcpy(conf_path, argv[1]);
  for (int i = 2; i < _argc; i++) {
    argv[i - 1] = argv[i];
  }
  argc = _argc - 1;

  ret = runtime_init(conf_path, _main, argv);
  if (ret) {
    std::cerr << "failed to start runtime" << std::endl;
    return ret;
  }

  return 0;
}
