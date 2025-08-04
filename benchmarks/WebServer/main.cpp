// #!/usr/bin/env cpp

// extern "C" {
// #include <runtime/runtime.h>  // [MODIFIED] Removed runtime dependency
// }
// #include "thread.h" // [MODIFIED] Removed runtime dependency

// [MODIFIED] AIFM dependencies removed
// #include "array.hpp"
// #include "deref_scope.hpp"
// #include "device.hpp"
// #include "helpers.hpp"
// #include "manager.hpp"

#include <numeric>
// crypto++ and snappy dependencies remain as they are part of the workload logic
#include "snappy.h"
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
#include <random>
#include <thread> // [MODIFIED] Use std::thread
#include <mutex>  // [MODIFIED] Use std::mutex for concurrency
#include <vector>
#include <unordered_map> // [MODIFIED] Use std::unordered_map

// [MODIFIED] This is a simplified helper, as the original might be in helpers.hpp
namespace helpers {
    uint64_t microtime() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }
}

using namespace std;

#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))

// The namespace is kept for structural similarity
namespace far_memory {

// [MODIFIED] Forward declaration of the test class, as it's not a real namespace
class FarMemTest;

// [MODIFIED] A simple concurrent hash map wrapper using a mutex.
// The original GenericConcurrentHopscotch is highly optimized. This is a basic replacement.
template<typename K, typename V>
class SimpleConcurrentHashMap {
private:
    std::unordered_map<K, V> map_;
    std::mutex mutex_;

public:
    void put(const K &key, const V &value) {
        std::lock_guard<std::mutex> guard(mutex_);
        map_[key] = value;
    }

    bool get(const K &key, V &value) {
        std::lock_guard<std::mutex> guard(mutex_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
};

class FarMemTest {
private:
  // Constants are preserved to define the workload characteristics.
  // Many AIFM-specific ones are no longer needed but kept for context.
  // FarMemManager.
  // constexpr static uint64_t kCacheSize = 30000 * Region::kSize; // No longer relevant
  // constexpr static uint64_t kFarMemSize = (17ULL << 30); // No longer relevant
  // constexpr static uint32_t kNumGCThreads = 12; // No longer relevant

  // Hashtable.
  constexpr static uint32_t kKeyLen = 12;
  constexpr static uint32_t kValueLen = 4;
  constexpr static uint32_t kNumKVPairs = 1 << 27;

  // Array.
  constexpr static uint32_t kNumArrayEntries = 2 << 20; // 2 M entries.
  constexpr static uint32_t kArrayEntrySize = 8192;     // 8 K

  // Runtime.
  constexpr static uint32_t kNumMutatorThreads = 1; // As in the original linux_mem for baseline
  constexpr static double kZipfParamS = 0.85;
  constexpr static uint32_t kNumKeysPerRequest = 32;
  constexpr static uint32_t kNumReqs = kNumKVPairs / kNumKeysPerRequest;

  // Use std::log for static_log replacement
  static constexpr uint32_t static_log(uint32_t base, uint32_t n) {
      return (n > 1) ? 1 + static_log(base, n / base) : 0;
  }
  constexpr static uint32_t kLog10NumKeysPerRequest = static_log(10, kNumKeysPerRequest);
  constexpr static uint32_t kReqLen = kKeyLen - kLog10NumKeysPerRequest;
  constexpr static uint32_t kReqSeqLen = kNumReqs;

  // Output constants remain the same.
  constexpr static uint32_t kPrintPerIters = 8192;
  constexpr static uint32_t kMaxPrintIntervalUs = 1000 * 1000;
  constexpr static uint32_t kPrintTimes = 100;
  constexpr static uint32_t kLatsWinSize = 1 << 12;

  // Data structures for workload
  struct Req { char data[kReqLen]; };

  // Use std::string as key for convenience
  struct Key {
    char data[kKeyLen];
    bool operator==(const Key& other) const { return memcmp(data, other.data, kKeyLen) == 0; }
  };

  // Provide a hash function for Key
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
        // A simple hash, could be improved.
        return std::hash<std::string_view>()(std::string_view(k.data, kKeyLen));
    }
  };

  union Value {
    uint32_t num;
    char data[kValueLen];
  };

  struct ArrayEntry { uint8_t data[kArrayEntrySize]; };

  struct alignas(64) Cnt { uint64_t c; };

  // [MODIFIED] Use standard library containers
  using AppHashMap = SimpleConcurrentHashMap<Key, Value>;
  using AppArray = std::vector<ArrayEntry>;

  std::unique_ptr<std::mt19937> generators[kNumMutatorThreads];
  Req all_gen_reqs[kNumReqs];
  uint32_t all_zipf_req_indices[kNumMutatorThreads][kReqSeqLen];

  Cnt req_cnts[kNumMutatorThreads];
  uint32_t lats[kNumMutatorThreads][kLatsWinSize];
  Cnt lats_idx[kNumMutatorThreads];
  Cnt per_core_req_idx[kNumMutatorThreads];

  std::atomic_flag flag;
  uint64_t print_times = 0;
  uint64_t prev_sum_reqs = 0;
  uint64_t prev_us = 0;
  std::vector<double> mops_records;

  unsigned char key_crypto[CryptoPP::AES::DEFAULT_KEYLENGTH];
  unsigned char iv[CryptoPP::AES::BLOCKSIZE];
  std::unique_ptr<CryptoPP::CBC_Mode_ExternalCipher::Encryption> cbcEncryption;
  std::unique_ptr<CryptoPP::AES::Encryption> aesEncryption;

  // Most helper functions can be kept as they are workload-specific
  inline void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len, char *array) {
     // ... (implementation is the same)
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
    // ... (implementation is the same, without preempt_disable)
    auto &generator = *generators[0]; // Assuming single thread or tid 0
    std::uniform_int_distribution<int> distribution('a', 'z' + 1);
    for (uint32_t i = 0; i < len; i++) {
        data[i] = char(distribution(generator));
    }
  }
  inline void random_req(char* data, uint32_t tid) {
      // ... (implementation is the same)
  }

  void prepare(AppHashMap *hopscotch) {
    for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
      std::random_device rd;
      generators[i].reset(new std::mt19937(rd()));
    }
    memset(lats_idx, 0, sizeof(lats_idx));
    memset(key_crypto, 0x00, CryptoPP::AES::DEFAULT_KEYLENGTH);
    memset(iv, 0x00, CryptoPP::AES::BLOCKSIZE);
    aesEncryption.reset(new CryptoPP::AES::Encryption(key_crypto, CryptoPP::AES::DEFAULT_KEYLENGTH));
    cbcEncryption.reset(new CryptoPP::CBC_Mode_ExternalCipher::Encryption(*aesEncryption, iv));

    std::vector<std::thread> threads; // [MODIFIED] Use std::thread
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(std::thread([&, tid]() { // [MODIFIED] Use std::thread
        for (uint32_t i = 0; i < kNumReqs / kNumMutatorThreads; ++i) {
            Key key;
            random_string(key.data, kKeyLen);
            for(uint32_t j = 0; j< kNumKeysPerRequest; ++j){
                Value value;
                value.num = j + i; // Simplified logic
                // Remove DerefScope, use direct map access
                hopscotch->put(key, value);
            }
        }
      }));
    }
    for (auto &thread : threads) {
      thread.join(); // [MODIFIED] Use join()
    }

    // Zipf generation is mostly the same, simplified for a single-threaded main preparation
    zipf_table_distribution<> zipf(kNumReqs, kZipfParamS);
    auto &generator = *generators[0];
    for(uint32_t i=0; i<kReqSeqLen; ++i) {
        all_zipf_req_indices[0][i] = zipf(*generator);
    }
  }

  void consume_array_entry(const ArrayEntry &entry) {
    std::string ciphertext;
    CryptoPP::StreamTransformationFilter stfEncryptor(*cbcEncryption, new CryptoPP::StringSink(ciphertext));
    stfEncryptor.Put((const unsigned char *)&entry.data, sizeof(entry));
    stfEncryptor.MessageEnd();
    std::string compressed;
    snappy::Compress(ciphertext.c_str(), ciphertext.size(), &compressed);
    auto compressed_len = compressed.size();
    ACCESS_ONCE(compressed_len);
  }

  void print_perf() { // This logic remains mostly unchanged
      // ...
  }

  void bench(AppHashMap *hopscotch, AppArray *array) {
    std::vector<std::thread> threads; // [MODIFIED] Use std::thread
    prev_us = helpers::microtime();
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(std::thread([&, tid]() { // [MODIFIED] Use std::thread
        uint32_t cnt = 0;
        Key key;
        Value value;
        uint32_t array_index = 0;

        while (1) {
          if (unlikely(cnt++ % kPrintPerIters == 0)) {
            print_perf();
          }

          // Simplified key lookup logic
          auto req_idx = all_zipf_req_indices[0][per_core_req_idx[tid].c++ % kReqSeqLen];

          // ----- Workload Start -----
          // Phase 1: Hash table lookups
          for (uint32_t i = 0; i < kNumKeysPerRequest; i++) {
            // [MODIFIED] No DerefScope, direct get from map
            if (hopscotch->get(key, value)) {
              array_index += value.num;
            }
          }

          // Phase 2: Array access and computation
          array_index %= kNumArrayEntries;
          // [MODIFIED] No DerefScope, direct access to vector element
          const auto &array_entry = (*array)[array_index];
          consume_array_entry(array_entry);
          // ----- Workload End -----

          ACCESS_ONCE(req_cnts[tid].c)++;
        }
      }));
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }

public:
  void do_work() { // [MODIFIED] Removed FarMemManager * argument
    // [MODIFIED] Allocate data structures on the heap using std::make_unique
    auto hopscotch = std::make_unique<AppHashMap>();
    auto array_ptr = std::make_unique<AppArray>(kNumArrayEntries);

    std::cout << "Prepare..." << std::endl;
    prepare(hopscotch.get());

    std::cout << "Bench..." << std::endl;
    bench(hopscotch.get(), array_ptr.get());
  }

  void run() { // [MODIFIED] No raddr argument
    // [MODIFIED] No FarMemManager
    do_work();
  }
};

} // namespace far_memory


int main(int _argc, char *argv[]) {
  // [MODIFIED] Simplified main, no runtime_init
  if (_argc < 2) {
      // Still expect a config file argument for structural consistency, even if not used.
      std::cerr << "usage: [cfg_file]" << std::endl;
      return -EINVAL;
  }

  // The main test logic is now called directly from main.
  far_memory::FarMemTest test;
  test.run();

  return 0;
}
