#include <algorithm>
#include <array>
#include <atomic>
#include <cassert> // Replaces BUG_ON
#include <chrono>
#include <cmath> // For std::pow in Zipfian
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>   // For std::mutex
#include <numeric>
#include <random> // For random number generation and distributions
#include <string>  // For std::string keys/values
#include <thread>  // For std::thread
#include <unordered_map>
#include <vector>

// Keep external libraries
#include "snappy.h"
#include "cryptopp/aes.h"
#include "cryptopp/filters.h"
#include "cryptopp/modes.h"


namespace far_memory { // Keeping the namespace, but it no longer implies far memory

// --- BEGIN FarMemory-specific helper/concept replacements ---

// FIX 1: Make the function constexpr so it can be used for compile-time constants.
static constexpr uint32_t static_log10(uint32_t n) {
    if (n == 0) return 1;
    uint32_t log = 0;
    uint32_t temp = n;
    while (temp > 0) {
        temp /= 10;
        log++;
    }
    return log;
}

// Minimalistic implementation for 'finally' guard (optional but useful)
template <typename F>
struct ScopeGuard {
    F f;
    explicit ScopeGuard(F f) noexcept : f(std::move(f)) {}
    ~ScopeGuard() { f(); }
};

template <typename F>
ScopeGuard<F> make_scope_guard(F f) noexcept {
    return ScopeGuard<F>(std::move(f));
}

// Replacement for microtime and rdtsc using std::chrono
namespace MyClock {
    inline uint64_t microtime() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    }
    inline uint64_t rdtsc_cycles() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    }
} // namespace MyClock

// Replacement for FarMemory's ConcurrentHopscotch (simplified to std::unordered_map with mutex)
class SynchronizedMap {
private:
    std::unordered_map<std::string, std::string> map_;
    std::mutex mutex_;

public:
    void put(uint32_t key_len, const uint8_t *key_data, uint32_t value_len, const uint8_t *value_data) {
        std::lock_guard<std::mutex> lock(mutex_);
        map_[std::string((const char*)key_data, key_len)] = std::string((const char*)value_data, value_len);
    }

    bool get(uint32_t key_len, const uint8_t *key_data, uint16_t *value_len_out, uint8_t *value_data_out) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(std::string((const char*)key_data, key_len));
        if (it != map_.end()) {
            uint32_t len_to_copy = std::min((uint32_t)it->second.length(), (uint32_t)*value_len_out);
            memcpy(value_data_out, it->second.data(), len_to_copy);
            *value_len_out = len_to_copy;
            return true;
        }
        return false;
    }
};

// Simple Zipfian distribution generator
class ZipfianGenerator {
private:
    std::mt19937 *generator_;
    std::discrete_distribution<uint32_t> distribution_;

public:
    ZipfianGenerator(uint32_t num_items, double skew, std::mt19937 *gen)
        : generator_(gen) {
        std::vector<double> weights(num_items);
        for (uint32_t i = 0; i < num_items; ++i) {
            weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), skew);
        }
        distribution_ = std::discrete_distribution<uint32_t>(weights.begin(), weights.end());
    }

    uint32_t operator()() {
        return distribution_(*generator_);
    }
};

// --- END FarMemory-specific helper/concept replacements ---


class FarMemTest {
private:
  constexpr static uint32_t kKeyLen = 12;
  constexpr static uint32_t kValueLen = 4;
  constexpr static uint32_t kNumKVPairs = 1 << 27;

  constexpr static uint32_t kNumArrayEntries = 2 << 20; // 2 M entries.
  constexpr static uint32_t kArrayEntrySize = 8192;     // 8 K

  constexpr static uint32_t kNumMutatorThreads = 4;
  constexpr static double kZipfParamS = 0.85;
  constexpr static uint32_t kNumKeysPerRequest = 32;
  constexpr static uint32_t kNumReqs = kNumKVPairs / kNumKeysPerRequest;
  constexpr static uint32_t kLog10NumKeysPerRequest =
      static_log10(kNumKeysPerRequest);
  // This is now a valid compile-time constant thanks to constexpr static_log10
  constexpr static uint32_t kReqLen = kKeyLen - kLog10NumKeysPerRequest;
  constexpr static uint32_t kReqSeqLen = kNumReqs;

  constexpr static uint32_t kPrintPerIters = 8192;
  constexpr static uint32_t kMaxPrintIntervalUs = 1000 * 1000;
  constexpr static uint32_t kPrintTimes = 10;
  constexpr static uint32_t kLatsWinSize = 1 << 12;

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

  struct alignas(64) AtomicCnt {
    std::atomic<uint64_t> c;
    AtomicCnt() : c(0) {}
  };

  using AppArray = std::vector<ArrayEntry>;

  std::vector<std::unique_ptr<std::mt19937>> generators;
  Req* all_gen_reqs = new Req[kNumReqs]; // Heap allocate to avoid stack overflow
  uint32_t* all_zipf_req_indices[kNumMutatorThreads];

  AtomicCnt req_cnts[kNumMutatorThreads];
  uint32_t* lats[kNumMutatorThreads];
  AtomicCnt lats_idx[kNumMutatorThreads];
  AtomicCnt per_core_req_idx[kNumMutatorThreads];

  std::atomic_flag print_flag_lock = ATOMIC_FLAG_INIT;
  uint64_t print_times = 0;
  uint64_t prev_sum_reqs = 0;
  uint64_t prev_us = 0;
  std::vector<double> mops_records;

  unsigned char key[CryptoPP::AES::DEFAULT_KEYLENGTH];
  unsigned char iv[CryptoPP::AES::BLOCKSIZE];
  std::unique_ptr<CryptoPP::CBC_Mode_ExternalCipher::Encryption> cbcEncryption;
  std::unique_ptr<CryptoPP::AES::Encryption> aesEncryption;

public:
  // Constructor to correctly initialize heap-allocated arrays
  FarMemTest() {
      for(int i = 0; i < kNumMutatorThreads; ++i) {
          all_zipf_req_indices[i] = new uint32_t[kReqSeqLen];
          lats[i] = new uint32_t[kLatsWinSize];
          // Zero-initialize them
          memset(lats[i], 0, kLatsWinSize * sizeof(uint32_t));
          memset(all_zipf_req_indices[i], 0, kReqSeqLen * sizeof(uint32_t));
      }
  }

  // Destructor to free heap-allocated memory
  ~FarMemTest() {
      delete[] all_gen_reqs;
      for(int i = 0; i < kNumMutatorThreads; ++i) {
          delete[] all_zipf_req_indices[i];
          delete[] lats[i];
      }
  }

private:
  // FIX 5: Added a fix for the n=0 edge case.
  inline void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len,
                                          char *array) {
    uint32_t len = 0;
    if (n == 0 && suffix_len > 0) {
        array[len++] = '0';
    } else {
        uint32_t temp = n;
        while (temp > 0) {
            array[len++] = (temp % 10) + '0';
            temp /= 10;
        }
    }

    while (len < suffix_len) {
      array[len++] = '0';
    }
    std::reverse(array, array + suffix_len);
  }

  inline void random_string(char *data, uint32_t len, uint32_t tid) {
    assert(len > 0);
    auto &generator = *generators[tid];
    std::uniform_int_distribution<int> distribution('a', 'z' + 1);
    for (uint32_t i = 0; i < len; i++) {
      data[i] = char(distribution(generator));
    }
  }

  inline void random_req(char *data, uint32_t tid) {
    auto tid_len = static_log10(kNumMutatorThreads);
    random_string(data, kReqLen - tid_len, tid);
    append_uint32_to_char_array(tid, tid_len, data + kReqLen - tid_len);
  }

  inline uint32_t random_uint32(uint32_t tid) {
    auto &generator = *generators[tid];
    std::uniform_int_distribution<uint32_t> distribution(
        0, std::numeric_limits<uint32_t>::max());
    return distribution(generator);
  }

  // FIX 2: Removed unused 'array' parameter.
  void prepare(SynchronizedMap *hopscotch) {
    generators.reserve(kNumMutatorThreads);
    for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
        std::random_device rd;
        generators.emplace_back(std::make_unique<std::mt19937>(rd()));
    }

    // lats_idx is an array of AtomicCnt, which have default constructors.
    // No explicit memset needed if we rely on constructors.

    memset(key, 0x00, CryptoPP::AES::DEFAULT_KEYLENGTH);
    memset(iv, 0x00, CryptoPP::AES::BLOCKSIZE);

    aesEncryption.reset(
        new CryptoPP::AES::Encryption(key, CryptoPP::AES::DEFAULT_KEYLENGTH));
    cbcEncryption.reset(
        new CryptoPP::CBC_Mode_ExternalCipher::Encryption(*aesEncryption, iv));

    std::vector<std::thread> threads;
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(std::thread([&, tid]() {
        auto num_reqs_per_thread = kNumReqs / kNumMutatorThreads;
        auto req_offset = tid * num_reqs_per_thread;
        Req *thread_gen_reqs = &all_gen_reqs[req_offset];

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
            hopscotch->put(kKeyLen, (const uint8_t *)key.data, kValueLen,
                           (uint8_t *)value.data);
          }
          thread_gen_reqs[i] = req;
        }
      }));
    }
    for (auto &thread : threads) {
      thread.join();
    }

    std::cout << "Generating Zipfian distribution..." << std::endl;
    ZipfianGenerator zipf(kNumReqs, kZipfParamS, generators[0].get());
    constexpr uint32_t kPerCoreWinInterval = kReqSeqLen / kNumMutatorThreads;

    for (uint32_t i = 0; i < kReqSeqLen; i++) {
      auto rand_idx = zipf();
      for (uint32_t j = 0; j < kNumMutatorThreads; j++) {
        all_zipf_req_indices[j][(i + (j * kPerCoreWinInterval)) % kReqSeqLen] =
            rand_idx;
      }
    }
    std::cout << "Zipfian distribution generated." << std::endl;
  }

  void prepare_array(AppArray *array) {
    std::cout << "Preparing Array (resizing and filling placeholder data)..." << std::endl;
    array->resize(kNumArrayEntries);
    for(uint32_t i = 0; i < kNumArrayEntries; ++i) {
        memset(array->at(i).data, 'A' + (i % 26), kArrayEntrySize);
    }
    std::cout << "Array prepared." << std::endl;
  }

  void consume_array_entry(const ArrayEntry &entry) {
    std::string ciphertext;
    std::string compressed;
    CryptoPP::StreamTransformationFilter stfEncryptor(
        *cbcEncryption, new CryptoPP::StringSink(ciphertext));
    stfEncryptor.Put((const unsigned char *)&entry.data, sizeof(entry));
    stfEncryptor.MessageEnd();

    snappy::Compress(ciphertext.c_str(), ciphertext.size(), &compressed);
    volatile auto compressed_len = compressed.size();
    (void)compressed_len;
  }

  void print_perf() {
    if (!print_flag_lock.test_and_set(std::memory_order_acquire)) {
        auto guard = make_scope_guard([&]() { print_flag_lock.clear(std::memory_order_release); });

        auto us = MyClock::microtime();
        uint64_t sum_reqs = 0;
        for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
            sum_reqs += req_cnts[i].c.load(std::memory_order_relaxed);
        }
        if (us - prev_us > kMaxPrintIntervalUs) {
            auto mops =
                ((double)(sum_reqs - prev_sum_reqs) / (us - prev_us)); // Removed arbitrary multiplier
            mops_records.push_back(mops);
            if (print_times++ >= kPrintTimes) {
              constexpr double kRatioChosenRecords = 0.1;
              uint32_t num_chosen_records = mops_records.size() * kRatioChosenRecords;
              if (mops_records.size() > num_chosen_records) {
                mops_records.erase(mops_records.begin(),
                                   mops_records.end() - num_chosen_records);
              }

              if (!mops_records.empty()) {
                  std::cout << "mops = "
                            << std::accumulate(mops_records.begin(), mops_records.end(), 0.0) /
                                   mops_records.size()
                            << std::endl;
              }

              std::vector<uint32_t> all_lats;
              all_lats.reserve(kNumMutatorThreads * kLatsWinSize);
              for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
                auto num_lats = std::min((uint64_t)kLatsWinSize, lats_idx[i].c.load(std::memory_order_relaxed));
                all_lats.insert(all_lats.end(), &lats[i][0], &lats[i][num_lats]);
              }
              if (!all_lats.empty()) {
                  std::sort(all_lats.begin(), all_lats.end());
                  // Guard against empty vector
                  size_t p90_idx = static_cast<size_t>(all_lats.size() * 0.90);
                  if (p90_idx >= all_lats.size()) p90_idx = all_lats.size() - 1;
                  std::cout << "90 tail lat (chrono ns) = "
                            << all_lats[p90_idx] << std::endl;
              }
              exit(0);
            }
            prev_us = us;
            prev_sum_reqs = sum_reqs;
        }
    }
  }

  void bench(SynchronizedMap *hopscotch, AppArray *array) {
    std::vector<std::thread> threads;
    prev_us = MyClock::microtime();

    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(std::thread([&, tid]() {
        uint32_t cnt = 0;

        while (1) {
          if (cnt++ % kPrintPerIters == 0) {
            print_perf();
          }

          auto local_req_idx_ptr = &per_core_req_idx[tid].c;
          uint64_t current_req_idx = local_req_idx_ptr->load(std::memory_order_relaxed);
          auto req_idx = all_zipf_req_indices[tid][current_req_idx];
          if (local_req_idx_ptr->fetch_add(1, std::memory_order_relaxed) + 1 == kReqSeqLen) {
            local_req_idx_ptr->store(0, std::memory_order_relaxed);
          }

          // FIX 3: Typo `Request` corrected to `Req`.
          Req req = all_gen_reqs[req_idx];

          Key key;
          memcpy(key.data, req.data, kReqLen);
          auto start = MyClock::rdtsc_cycles();

          uint32_t array_index = 0;
          {
            for (uint32_t i = 0; i < kNumKeysPerRequest; i++) {
              append_uint32_to_char_array(i, kLog10NumKeysPerRequest,
                                          key.data + kReqLen);
              Value value;
              uint16_t value_len = kValueLen;
              if (hopscotch->get(kKeyLen, (const uint8_t *)key.data,
                                 &value_len, (uint8_t *)value.data)) {
                array_index += value.num;
              }
            }
          }
          {
            array_index %= kNumArrayEntries;
            const auto &array_entry = array->at(array_index);
            consume_array_entry(array_entry);
          }
          auto end = MyClock::rdtsc_cycles();

          lats[tid][(lats_idx[tid].c.fetch_add(1, std::memory_order_relaxed)) % kLatsWinSize] = static_cast<uint32_t>(end - start);
          req_cnts[tid].c.fetch_add(1, std::memory_order_relaxed);
        }
      }));
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }

public:
  void do_work() {
    auto hopscotch = std::make_unique<SynchronizedMap>();
    std::cout << "Prepare Hopscotch (simulated)..." << std::endl;
    prepare(hopscotch.get()); // Calling the corrected function

    auto array_ptr = std::make_unique<AppArray>();
    std::cout << "Prepare Array..." << std::endl;
    prepare_array(array_ptr.get());

    std::cout << "Benchmarking..." << std::endl;
    bench(hopscotch.get(), array_ptr.get());
  }

};
} // namespace far_memory

// Standard C++ main function
// FIX 4: Used [[maybe_unused]] to silence warnings about unused parameters.
int main(int [[maybe_unused]] argc, char *[[maybe_unused]] argv[]) {
  std::cout << "Starting FarMem Test (Simulated Standard C++ Version)..." << std::endl;

  // Note: Some arrays are very large and may cause a stack overflow if declared as
  // global/static variables in main's scope. Encapsulating in a class that
  // heap-allocates (or using std::vector/unique_ptr) is safer.
  // I moved the large arrays to be heap-allocated in the constructor.
  auto test = std::make_unique<far_memory::FarMemTest>();
  test->do_work();

  std::cout << "Benchmark complete." << std::endl;
  return 0;
}
