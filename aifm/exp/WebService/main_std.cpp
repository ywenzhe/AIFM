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
#include <thread>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
#include <cmath>

// 保留的非标准库
#include "snappy.h"
#include "cryptopp/aes.h"
#include "cryptopp/filters.h"
#include "cryptopp/modes.h"

#define ACCESS_ONCE(x) (x)

// 简化的Zipf分布实现
template <class IntType = unsigned long, class RealType = double>
class zipf_table_distribution {
public:
    typedef IntType result_type;
    
    zipf_table_distribution(const IntType n, const RealType s = 1.0) : n_(n), s_(s) {
        // 计算概率分布
        std::vector<RealType> weights(n);
        RealType sum = 0.0;
        for (IntType i = 1; i <= n; ++i) {
            weights[i-1] = 1.0 / std::pow(i, s);
            sum += weights[i-1];
        }
        // 归一化
        for (auto& w : weights) {
            w /= sum;
        }
        dist_ = std::discrete_distribution<IntType>(weights.begin(), weights.end());
    }
    
    IntType operator()(std::mt19937& rng) {
        return dist_(rng);
    }
    
    result_type min() const { return 0; }
    result_type max() const { return n_ - 1; }

private:
    IntType n_;
    RealType s_;
    std::discrete_distribution<IntType> dist_;
};

// 简化的并发哈希表实现
template<typename K, typename V>
class ConcurrentHashTable {
private:
    static constexpr size_t NUM_BUCKETS = 1024 * 1024; // 1M buckets，支持大规模数据
    
    struct Bucket {
        std::mutex mutex;
        std::unordered_map<K, V> map;
    };
    
    std::vector<Bucket> buckets_;
    
    size_t hash_key(const K& key) const {
        return std::hash<K>{}(key) % NUM_BUCKETS;
    }
    
public:
    ConcurrentHashTable() : buckets_(NUM_BUCKETS) {}
    
    void put(const K& key, const V& value) {
        size_t bucket_idx = hash_key(key);
        std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);
        buckets_[bucket_idx].map[key] = value;
    }
    
    bool get(const K& key, V& value) {
        size_t bucket_idx = hash_key(key);
        std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);
        auto it = buckets_[bucket_idx].map.find(key);
        if (it != buckets_[bucket_idx].map.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
};

// 简化的数组实现
template<typename T, size_t N>
class Array {
private:
    std::vector<T> data_;
    
public:
    Array() : data_(N) {}
    
    const T& at(size_t index) const {
        return data_[index % N];
    }
    
    T& at(size_t index) {
        return data_[index % N];
    }
    
    void disable_prefetch() {}
    void enable_prefetch() {}
};

class WebServiceBenchmark {
private:
    // 配置参数 - 大规模数据以测试内存限制影响
    static constexpr uint32_t kKeyLen = 12;
    static constexpr uint32_t kValueLen = 4;
    // 10GB hashmap: 假设每个KV对平均占用40字节 (key+value+overhead)
    // 10GB / 40 bytes ≈ 268M entries
    static constexpr uint64_t kNumKVPairs = 268ULL << 20; // 268M entries ≈ 10GB
    // 16GB array: 16GB / 8KB = 2M entries
    static constexpr uint32_t kNumArrayEntries = (16ULL << 30) / 8192; // 2M entries = 16GB
    static constexpr uint32_t kArrayEntrySize = 8192; // 8K
    static constexpr uint32_t kNumMutatorThreads = 8;
    static constexpr double kZipfParamS = 0.85;
    static constexpr uint32_t kNumKeysPerRequest = 32;
    static constexpr uint64_t kNumReqs = kNumKVPairs / kNumKeysPerRequest;
    static constexpr uint32_t kReqLen = kKeyLen - 2;
    static constexpr uint64_t kReqSeqLen = kNumReqs;
    static constexpr uint32_t kPrintPerIters = 8192;
    static constexpr uint32_t kMaxPrintIntervalUs = 1000 * 1000; // 1 second
    static constexpr uint32_t kPrintTimes = 100;
    
    uint32_t kNumCPUs;
    
    struct Req {
        char data[kReqLen];
    };
    
    struct Key {
        char data[kKeyLen];
        
        bool operator==(const Key& other) const {
            return memcmp(data, other.data, kKeyLen) == 0;
        }
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
    using HashTable = ConcurrentHashTable<std::string, Value>;
    
    // 成员变量
    std::vector<std::unique_ptr<std::mt19937>> generators;
    std::vector<Req> all_gen_reqs;
    std::vector<std::vector<uint64_t>> all_zipf_req_indices;  // 使用uint64_t支持更大索引
    std::vector<Cnt> req_cnts;
    std::vector<Cnt> local_array_miss_cnts;
    std::vector<Cnt> local_hashtable_miss_cnts;
    std::vector<Cnt> per_core_req_idx;
    
    std::atomic<bool> flag{false};
    uint64_t print_times = 0;
    uint64_t prev_sum_reqs = 0;
    uint64_t prev_sum_array_misses = 0;
    uint64_t prev_sum_hashtable_misses = 0;
    uint64_t prev_us = 0;
    uint64_t running_us = 0;
    uint64_t start_us = 0;
    std::atomic<bool> should_stop{false};
    double runtime_seconds = 0.0;
    
    // 加密相关
    unsigned char key[CryptoPP::AES::DEFAULT_KEYLENGTH];
    unsigned char iv[CryptoPP::AES::BLOCKSIZE];
    std::unique_ptr<CryptoPP::CBC_Mode_ExternalCipher::Encryption> cbcEncryption;
    std::unique_ptr<CryptoPP::AES::Encryption> aesEncryption;
    
    uint64_t microtime() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }
    
    void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len, char* array) {
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
    
    void random_string(char* data, uint32_t len, std::mt19937& generator) {
        std::uniform_int_distribution<int> distribution('a', 'z');
        for (uint32_t i = 0; i < len; i++) {
            data[i] = char(distribution(generator));
        }
    }
    
    void random_req(char* data, uint32_t tid, std::mt19937& generator) {
        auto tid_len = 2; // 简化
        random_string(data, kReqLen - tid_len, generator);
        append_uint32_to_char_array(tid, tid_len, data + kReqLen - tid_len);
    }
    
    void prepare(HashTable* hashtable) {
        std::cout << "Initializing random generators..." << std::endl;
        // 初始化随机数生成器
        generators.resize(kNumCPUs);
        for (uint32_t i = 0; i < kNumCPUs; i++) {
            std::random_device rd;
            generators[i] = std::make_unique<std::mt19937>(rd());
        }
        
        std::cout << "Initializing encryption..." << std::endl;
        // 初始化加密
        memset(key, 0x00, CryptoPP::AES::DEFAULT_KEYLENGTH);
        memset(iv, 0x00, CryptoPP::AES::BLOCKSIZE);
        aesEncryption = std::make_unique<CryptoPP::AES::Encryption>(key, CryptoPP::AES::DEFAULT_KEYLENGTH);
        cbcEncryption = std::make_unique<CryptoPP::CBC_Mode_ExternalCipher::Encryption>(*aesEncryption, iv);
        
        std::cout << "Generating " << kNumReqs << " requests and " << kNumKVPairs << " KV pairs..." << std::endl;
        std::cout << "Expected hashmap size: ~10GB, Array size: ~16GB" << std::endl;
        
        // 生成请求数据和填充哈希表
        all_gen_reqs.resize(kNumReqs);
        std::vector<std::thread> threads;
        
        for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
            threads.emplace_back([&, tid]() {
                auto num_reqs_per_thread = kNumReqs / kNumMutatorThreads;
                auto req_offset = tid * num_reqs_per_thread;
                
                for (uint64_t i = 0; i < num_reqs_per_thread; i++) {
                    if (i % 100000 == 0) {
                        std::cout << "Thread " << tid << " processed " << i << " requests" << std::endl;
                    }
                    
                    Req req;
                    random_req(req.data, tid, *generators[tid % kNumCPUs]);
                    Key k;
                    memcpy(k.data, req.data, kReqLen);
                    
                    for (uint32_t j = 0; j < kNumKeysPerRequest; j++) {
                        append_uint32_to_char_array(j, 2, k.data + kReqLen);
                        std::string key_str(k.data, kKeyLen);
                        Value value;
                        value.num = (j ? 0 : req_offset + i);
                        hashtable->put(key_str, value);
                    }
                    all_gen_reqs[req_offset + i] = req;
                }
                std::cout << "Thread " << tid << " completed!" << std::endl;
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        std::cout << "Generating Zipf distribution indices..." << std::endl;
        // 生成Zipf分布的请求序列
        all_zipf_req_indices.resize(kNumCPUs);
        zipf_table_distribution<> zipf(kNumReqs, kZipfParamS);
        
        for (uint32_t i = 0; i < kNumCPUs; i++) {
            all_zipf_req_indices[i].resize(kReqSeqLen);
            for (uint64_t j = 0; j < kReqSeqLen; j++) {
                all_zipf_req_indices[i][j] = zipf(*generators[i]);
            }
            std::cout << "Generated Zipf indices for CPU " << i << std::endl;
        }
        std::cout << "Preparation completed!" << std::endl;
    }
    
    void prepare(AppArray* /* array */) {
        // 数组初始化（为了性能测试，这里不做任何操作）
    }
    
    void consume_array_entry(const ArrayEntry& entry) {
        std::string ciphertext;
        CryptoPP::StreamTransformationFilter stfEncryptor(
            *cbcEncryption, new CryptoPP::StringSink(ciphertext));
        stfEncryptor.Put((const unsigned char*)&entry.data, sizeof(entry));
        stfEncryptor.MessageEnd();
        
        std::string compressed;
        snappy::Compress(ciphertext.c_str(), ciphertext.size(), &compressed);
        volatile auto compressed_len = compressed.size();
        (void)compressed_len;  // 避免未使用变量警告
    }
    
    void print_perf() {
        bool expected = false;
        if (flag.compare_exchange_strong(expected, true)) {
            auto us = microtime();
            if (us - prev_us > kMaxPrintIntervalUs) {
                running_us += (us - prev_us);
                if (print_times++ >= kPrintTimes) {
                    runtime_seconds = (double)(microtime() - start_us) / 1e6;
                    should_stop.store(true, std::memory_order_relaxed);
                }
                prev_us = us;
            }
            flag.store(false);
        }
    }
    
    void bench(HashTable* hashtable, AppArray* array) {
        std::vector<std::thread> threads;
        prev_us = microtime();
        start_us = prev_us;
        
        for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
            threads.emplace_back([&, tid]() {
                uint32_t cnt = 0;
                uint32_t core_id = tid % kNumCPUs;
                
                while (!should_stop.load(std::memory_order_relaxed)) {
                    if (cnt++ % kPrintPerIters == 0) {
                        print_perf();
                    }
                    
                    auto req_idx = all_zipf_req_indices[core_id][per_core_req_idx[core_id].c];
                    if (++per_core_req_idx[core_id].c == kReqSeqLen) {
                        per_core_req_idx[core_id].c = 0;
                    }
                    
                    auto& req = all_gen_reqs[req_idx];
                    Key k;
                    memcpy(k.data, req.data, kReqLen);
                    uint32_t array_index = 0;
                    
                    // 哈希表访问
                    for (uint32_t i = 0; i < kNumKeysPerRequest; i++) {
                        append_uint32_to_char_array(i, 2, k.data + kReqLen);
                        std::string key_str(k.data, kKeyLen);
                        Value value{};  // 初始化为零
                        bool found = hashtable->get(key_str, value);
                        ACCESS_ONCE(local_hashtable_miss_cnts[tid].c) += !found;
                        array_index += value.num;
                    }
                    
                    // 数组访问
                    array_index %= kNumArrayEntries;
                    const auto& array_entry = array->at(array_index);
                    consume_array_entry(array_entry);
                    
                    ACCESS_ONCE(req_cnts[tid].c)++;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        std::cout << "runtime_seconds = " << runtime_seconds << std::endl;
    }
    
public:
    WebServiceBenchmark() : kNumCPUs(std::max(1u, std::thread::hardware_concurrency())) {
        req_cnts.resize(kNumMutatorThreads);
        local_array_miss_cnts.resize(kNumMutatorThreads);
        local_hashtable_miss_cnts.resize(kNumMutatorThreads);
        per_core_req_idx.resize(kNumCPUs);
        
        // 初始化计数器
        for (auto& cnt : req_cnts) cnt.c = 0;
        for (auto& cnt : local_array_miss_cnts) cnt.c = 0;
        for (auto& cnt : local_hashtable_miss_cnts) cnt.c = 0;
        for (auto& cnt : per_core_req_idx) cnt.c = 0;
    }
    
    void run() {
        auto hashtable = std::make_unique<HashTable>();
        std::cout << "Prepare..." << std::endl;
        prepare(hashtable.get());
        
        auto array_ptr = std::make_unique<AppArray>();
        array_ptr->disable_prefetch();
        prepare(array_ptr.get());
        
        std::cout << "Bench..." << std::endl;
        bench(hashtable.get(), array_ptr.get());
    }
};

// 内存限制函数
void set_memory_limit_mb(size_t limit_mb) {
    struct rlimit limit;
    limit.rlim_cur = limit_mb * 1024ULL * 1024ULL; // MB to bytes
    limit.rlim_max = limit_mb * 1024ULL * 1024ULL;
    
    if (setrlimit(RLIMIT_AS, &limit) != 0) {
        std::cerr << "Failed to set memory limit to " << limit_mb << " MB" << std::endl;
        perror("setrlimit");
        exit(1);
    }
    
    std::cout << "Memory limit set to " << limit_mb << " MB" << std::endl;
}

// 获取当前内存使用量
size_t get_memory_usage_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key, value, unit;
            iss >> key >> value >> unit;
            return std::stoull(value);
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <memory_limit_mb>" << std::endl;
        return -1;
    }
    
    size_t memory_limit_mb = std::stoull(argv[1]);
    set_memory_limit_mb(memory_limit_mb);
    
    std::cout << "Starting WebService benchmark with " << memory_limit_mb << " MB memory limit" << std::endl;
    
    // 记录初始内存使用量
    size_t initial_memory_kb = get_memory_usage_kb();
    std::cout << "Initial memory usage: " << initial_memory_kb << " KB" << std::endl;
    
    try {
        WebServiceBenchmark benchmark;
        benchmark.run();
        
        // 记录最终内存使用量
        size_t final_memory_kb = get_memory_usage_kb();
        std::cout << "Final memory usage: " << final_memory_kb << " KB" << std::endl;
        std::cout << "Memory usage delta: " << (final_memory_kb - initial_memory_kb) << " KB" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return 1;
    }
    
    return 0;
}
