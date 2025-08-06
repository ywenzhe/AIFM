#include <algorithm> // For std::reverse, std::min, std::sort
#include <array>     // For std::array (though vector is used more)
#include <atomic>    // For std::atomic_flag, std::atomic
#include <cassert>   // For assert (BUG_ON replacement)
#include <chrono>    // For std::chrono::high_resolution_clock
#include <cstdint>   // For fixed-width integer types
#include <cstdlib>   // For std::exit
#include <cstring>   // For memset, memcpy
#include <iostream>  // For std::cout, std::cerr
#include <limits>    // For std::numeric_limits
#include <memory>    // For std::unique_ptr
#include <numeric>   // For std::accumulate
#include <random>    // For std::mt19937, std::uniform_int_distribution, std::random_device
#include <string>    // For std::string
#include <thread>    // For std::thread, std::this_thread, std::thread::hardware_concurrency
#include <vector>    // For std::vector
#include <mutex>     // For std::mutex, std::lock_guard
#include <unordered_map> // For std::unordered_map
 
// Replace ACCESS_ONCE for volatile access (for non-atomic variables)
// This ensures the compiler does not optimize away loads/stores to such variables.
#define ACCESS_VOLATILE(x) (*(volatile decltype(x) *)&(x))
// Replace BUG_ON with a standard assert
#define BUG_ON(condition) assert(!(condition))
// Remove performance hints, they are compiler/runtime specific macros
#define unlikely(x) (x)
 
// Define helpers_static_log as a constexpr function, as `helpers.hpp` is removed.
namespace InternalHelpers {
static constexpr uint32_t static_log(uint32_t base, uint32_t val) {
    uint32_t count = 0;
    if (val == 0) return 0; // log(0) is undefined, just return 0 or assert for specific needs
    // Loop until val is less than base. Equivalently, count how many times base divides val.
    while (val >= base && val > 0) { // val > 0 check to prevent infinite loop for val = 0 (though handled above)
        val /= base;
        count++;
    }
    return count;
}
} // namespace InternalHelpers
 
 
// Original namespace far_memory. While "far_memory" is no longer applicable,
// keeping it for structural consistency with the original code.
namespace far_memory {
 
// Using a global constant for max cores/threads for array sizing,
// as `hardware_concurrency()` is runtime-dependent and fixed-size arrays need a compile-time constant.
// This is a trade-off for simplicity; a more flexible design would use `std::vector` of generators, etc.
// For the purpose of this example, assuming a reasonable upper bound.
constexpr static uint32_t kMaxLogicalCPUs = 256;
 
class FarMemTest {
private:
  // FarMemManager related constants become irrelevant.
  // Hashtable.
  constexpr static uint32_t kKeyLen = 12;
  constexpr static uint32_t kValueLen = 4;
  constexpr static uint32_t kNumKVPairs = 1 << 27;
 
  // Array.
  constexpr static uint32_t kNumArrayEntries = 2 << 20; // 2 M entries.
  constexpr static uint32_t kArrayEntrySize = 8192;     // 8 K
 
  // Runtime.
  // Using a fixed number of mutator threads for this example.
  // In original Shenango, this might be related to `runtime_init` args or other config.
  // For standard C++, it could be std::thread::hardware_concurrency() or user-defined.
  constexpr static uint32_t kNumMutatorThreads = 12; // Example: Set to reasonable count for multi-threading.
                                                      // Original was `1`, but makes more sense to test multi-thread.
  constexpr static double kZipfParamS = 0.85;
  constexpr static uint32_t kNumKeysPerRequest = 32;
  constexpr static uint32_t kNumReqs = kNumKVPairs / kNumKeysPerRequest;
  // Using our internal helpers for static_log
  constexpr static uint32_t kLog10NumKeysPerRequest = InternalHelpers::static_log(10, kNumKeysPerRequest);
  constexpr static uint32_t kReqLen = kKeyLen - kLog10NumKeysPerRequest;
  constexpr static uint32_t kReqSeqLen = kNumReqs;
 
  // Output.
  constexpr static uint32_t kPrintPerIters = 8192;
  constexpr static uint32_t kMaxPrintIntervalUs = 1000 * 1000; // 1 second(s).
  constexpr static uint32_t kPrintTimes = 100;
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
 
  // std::vector is used here for AppArray instead of a custom Array type.
  using AppArray = std::vector<ArrayEntry>;
 
  // Use std::vector for dynamic sizing based on actual hardware concurrency if preferred,
  // or a fixed-size array if kMaxLogicalCPUs is a safe upper bound.
  // Using fixed size array here due to previous code structure.
  std::unique_ptr<std::mt19937> generators[kMaxLogicalCPUs];
  std::mutex generator_mu[kMaxLogicalCPUs]; // Mutexes for each generator for thread-safe access
 
  // Aligned for cache line optimization
  alignas(64) Req all_gen_reqs[kNumReqs];
  // `all_zipf_req_indices` distributed by logical CPU/thread.
  // kMaxLogicalCPUs defines upper bound. Actual usage will be `tid % kMaxLogicalCPUs`.
  uint32_t all_zipf_req_indices[kMaxLogicalCPUs][kReqSeqLen];
  std::mutex zipf_indices_mu; // Mutex to protect global zipf table generation
 
  // Replaced `Cnt` struct with `std::atomic<uint64_t>` for true atomicity
  std::atomic<uint64_t> req_cnts[kNumMutatorThreads];
  // Latency storage now per-mutator-thread
  uint32_t lats[kNumMutatorThreads][kLatsWinSize];
  std::atomic<uint64_t> lats_idx[kNumMutatorThreads];
  std::atomic<uint64_t> per_core_req_idx[kNumMutatorThreads]; // Using this as per-thread request index
  std::mutex print_mu; // Mutex to protect `print_perf` and shared data accessed within it.
 
  std::atomic_flag flag = ATOMIC_FLAG_INIT; // Initialize atomic_flag
  uint64_t print_times = 0;
  uint64_t prev_sum_reqs = 0;
  uint64_t prev_us = 0;
  std::vector<double> mops_records;
 
  // Hash table replacement: std::unordered_map protected by a mutex
  // Using std::string as key because `char data[]` for keys can't be used directly as map keys.
  // Value stores an actual `Value` union.
  std::unordered_map<std::string, Value> hopscotch_map_std;
  std::mutex map_mu; // Mutex for protecting access to `hopscotch_map_std`
 
  /*
  *  inline helper functions
  */
  inline void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len,
                                          char *array) {
    BUG_ON(suffix_len == 0); // Ensure suffix_len is sensible
    if (suffix_len == 0) return;
 
    // Handle n=0 specifically to ensure '0' is printed correctly for suffix_len >= 1
    if (n == 0) {
        if (suffix_len > 0) array[0] = '0';
        for (uint32_t i = 1; i < suffix_len; ++i) array[i] = '0';
        return;
    }
 
    uint32_t current_len = 0;
    // Extract digits in reverse order
    uint32_t temp_n = n;
    while (temp_n > 0 && current_len < suffix_len) { // Also cap by suffix_len
        auto digit = temp_n % 10;
        array[current_len++] = digit + '0';
        temp_n = temp_n / 10;
    }
 
    // Pad with leading zeros if current_len is less than suffix_len
    while (current_len < suffix_len) {
        array[current_len++] = '0';
    }
    // Reverse only the part of the array that was filled for the number
    std::reverse(array, array + suffix_len);
  }
 
  // Modified to take tid for generator access
  inline void random_string(char *data, uint32_t len, uint32_t tid) {
    BUG_ON(len <= 0);
    // Use modulo `kMaxLogicalCPUs` to ensure tid maps to a valid generator array index.
    // This is a simplification, as tid doesn't perfectly map to core numbers in std::thread context.
    std::lock_guard<std::mutex> lock(generator_mu[tid % kMaxLogicalCPUs]);
    auto &generator = *generators[tid % kMaxLogicalCPUs];
    std::uniform_int_distribution<int> distribution('a', 'z' + 1);
    for (uint32_t i = 0; i < len; i++) {
        data[i] = char(distribution(generator));
    }
  }
 
  inline void random_req(char *data, uint32_t tid) {
    auto tid_len = InternalHelpers::static_log(10, kNumMutatorThreads);
    random_string(data, kReqLen - tid_len, tid);
    append_uint32_to_char_array(tid, tid_len, data + kReqLen - tid_len);
  }
 
  // Not used elsewhere in the provided snippet once non-standard parts are removed.
  inline uint32_t random_uint32(uint32_t tid) {
    std::lock_guard<std::mutex> lock(generator_mu[tid % kMaxLogicalCPUs]);
    auto &generator = *generators[tid % kMaxLogicalCPUs];
    std::uniform_int_distribution<uint32_t> distribution(
        0, std::numeric_limits<uint32_t>::max());
    return distribution(generator);
  }
 
  // Mock `GenericConcurrentHopscotch` methods using `std::unordered_map`.
  void hopscotch_put(const char* key_data, uint32_t key_len, const uint8_t* value_data, uint32_t value_len) {
      std::string key_str(key_data, key_len);
      Value value_obj;
      // Copy value data, ensuring not to overflow the `Value` union.
      memcpy(value_obj.data, value_data, std::min((uint32_t)sizeof(value_obj.data), value_len));
      std::lock_guard<std::mutex> lock(map_mu); // Protect map access
      hopscotch_map_std[key_str] = value_obj;
  }
 
  void hopscotch_get(const char* key_data, uint32_t key_len, uint16_t* value_len_out, uint8_t* value_data_out) {
      std::string key_str(key_data, key_len);
      std::lock_guard<std::mutex> lock(map_mu); // Protect map access
      auto it = hopscotch_map_std.find(key_str);
      if (it != hopscotch_map_std.end()) {
          *value_len_out = static_cast<uint16_t>(kValueLen); // Assuming kValueLen is the max value length
          memcpy(value_data_out, it->second.data, *value_len_out);
      } else {
          // Key not found: return 0 length and fill with zeros.
          *value_len_out = 0;
          std::memset(value_data_out, 0, kValueLen);
      }
  }
 
  // Prepares the hash table and zipf indices. Original `prepare(GenericConcurrentHopscotch*)`.
  void prepare_hopscotch() {
    // Initialize generators for all possible logical CPUs/threads
    for (uint32_t i = 0; i < kMaxLogicalCPUs; i++) {
      std::random_device rd; // Seed generator from a non-deterministic source
      generators[i].reset(new std::mt19937(rd()));
    }
 
    // Initialize atomic counters for all mutator threads
    for (uint32_t i = 0; i < kNumMutatorThreads; ++i) {
        req_cnts[i].store(0, std::memory_order_relaxed);
        lats_idx[i].store(0, std::memory_order_relaxed);
        per_core_req_idx[i].store(0, std::memory_order_relaxed);
    }
    // Clear latency array
    std::memset(lats, 0, sizeof(lats));
 
    std::vector<std::thread> threads;
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back([&, tid]() {
        auto num_reqs_per_thread = kNumReqs / kNumMutatorThreads;
        auto req_offset = tid * num_reqs_per_thread;
        // `thread_local_gen_reqs` points to the thread's slice of the global `all_gen_reqs` array
        Req *thread_local_gen_reqs = &all_gen_reqs[req_offset];
 
        for (uint32_t i = 0; i < num_reqs_per_thread; i++) {
          Req req;
          random_req(req.data, tid); // Pass tid for thread-local generator access within random_string
          Key key;
          std::memcpy(key.data, req.data, kReqLen); // Copy initial part of key from request
          for (uint32_t j = 0; j < kNumKeysPerRequest; j++) {
            // Append unique suffix to create distinct keys
            append_uint32_to_char_array(j, kLog10NumKeysPerRequest, key.data + kReqLen);
            Value value;
            value.num = (j ? 0 : req_offset + i); // Value based on request index (for the first key)
            hopscotch_put(key.data, kKeyLen, (uint8_t *)value.data, kValueLen);
            // After put, ensure key.data is reset for the next iteration if req.data is reused.
            // In this specific loop, `req.data` is copied into `key.data` at the beginning,
            // so `key.data` gets properly updated.
          }
          thread_local_gen_reqs[i] = req; // Store generated request (for later use in benchmark)
        }
      });
    }
 
    // Join all threads to ensure setup is complete before proceeding
    for (auto &thread : threads) {
      thread.join();
    }
 
    // Zipf table generation (now globally synchronized)
    // Synchronize to avoid race conditions during zipf table creation for all_zipf_req_indices
    std::lock_guard<std::mutex> zipf_lock(zipf_indices_mu);
    // zipf_table_distribution is assumed to be available (e.g., from an external library or custom implementation).
    // If unavailable, this part needs to be replaced with a standard random distribution.
    // For this example, assuming `zipf_table_distribution` and it's thread-safe for generation.
    // If `zipf_table_distribution` is from `zipf.hpp` originally, it's non-standard.
    // For a fully standard solution, one would need to implement Zipf distribution using std::random.
    
    // NOT REPLACEABLE (direct standard equivalent not available, requires custom implementation or external lib)
    // #include "zipf.hpp" (assuming this was the original source)
    
    // Dummy zipf distribution for compilation if zipf.hpp is not provided:
    // If you don't have zipf.hpp, uncomment this simple dummy `zipf` that just returns a random index.
    // This will *not* be a true Zipf distribution.
    class DummyZipfDistribution {
    public:
        DummyZipfDistribution(uint32_t max_val, double /* param_s */) : dist_(0, max_val - 1) {}
        uint32_t operator()(std::mt19937& gen) {
            return dist_(gen);
        }
    private:
        std::uniform_int_distribution<uint32_t> dist_;
    };
    DummyZipfDistribution zipf(kNumReqs, kZipfParamS);
    // End of DummyZipfDistribution. If you have zipf.hpp, comment out DummyZipfDistribution and uncomment #include "zipf.hpp" if it exists.
 
    // Use generator from thread 0 for creating global zipf indices for simplicity
    std::mt19937& global_generator_for_zipf = *generators[0]; 
 
    // Distribute indices across `kNumMutatorThreads` instead of original `kNumCPUs`
    constexpr uint32_t kPerThreadWinInterval = kReqSeqLen / kNumMutatorThreads;
    for (uint32_t i = 0; i < kReqSeqLen; i++) {
      auto rand_idx = zipf(global_generator_for_zipf);
      for (uint32_t j = 0; j < kNumMutatorThreads; j++) {
        // Distribute within each thread's section of the zipf indices array
        all_zipf_req_indices[j % kMaxLogicalCPUs][(i + (j * kPerThreadWinInterval)) % kReqSeqLen] = rand_idx;
      }
    }
  }
 
  // Prepares the array. Original `prepare(AppArray*)`.
  void prepare_array(AppArray *array) {
    // Resize the std::vector to `kNumArrayEntries`
    array->resize(kNumArrayEntries);
    // Elements are default-constructed. Initialization logic could go here if needed.
  }
 
  // Stub out consume_array_entry without Crypto++ and Snappy
  void consume_array_entry(const ArrayEntry &entry) {
    // NOT REPLACEABLE (standard C++ does not offer crypto or compression libs)
    // Original: encryption -> compression.
    // Now: a simple sum to simulate computational work and prevent compiler optimizing away the access.
    volatile uint64_t dummy_sum = 0; // Use volatile to ensure calculation is not optimized away
    for (size_t i = 0; i < sizeof(entry.data); ++i) {
        dummy_sum += entry.data[i];
    }
    // Ensure the dummy_sum is "used" to prevent entire loop optimization.
    ACCESS_VOLATILE(dummy_sum);
  }
 
  void print_perf() {
    // Only one thread at a time enters this critical section
    if (!flag.test_and_set(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(print_mu); // Protect shared print data
 
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch())
                    .count();
      uint64_t sum_reqs = 0;
      for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
        sum_reqs += req_cnts[i].load(std::memory_order_relaxed); // Use load() for atomic
      }
 
      if (us - prev_us > kMaxPrintIntervalUs) {
        // Multiplied by 1.098, possibly a calibration factor from original benchmark setup.
        auto mops = ((double)(sum_reqs - prev_sum_reqs) / (us - prev_us)) * 1.098;
        mops_records.push_back(mops);
        
        // Re-measure time after intensive calculations to ensure `prev_us` is fresh for next interval
        us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch())
                    .count();
 
        if (print_times++ >= kPrintTimes) {
          constexpr double kRatioChosenRecords = 0.1;
          uint32_t num_chosen_records =
              static_cast<uint32_t>(mops_records.size() * kRatioChosenRecords);
          
          // Ensure `num_chosen_records` is valid for `erase`
          if (num_chosen_records > 0 && num_chosen_records <= mops_records.size()) {
              mops_records.erase(mops_records.begin(),
                                 mops_records.end() - num_chosen_records);
          } else if (mops_records.size() > 0) { // If ratio results in 0 or too large, but records exist
              num_chosen_records = mops_records.size(); // Take all if few records, or if ratio too large
          }
 
          if (!mops_records.empty()) {
            std::cout << "mops = "
                      << std::accumulate(mops_records.begin(), mops_records.end(),
                                    0.0) / mops_records.size()
                      << std::endl;
          } else {
             std::cout << "mops = 0 (no records)" << std::endl;
          }
 
          std::vector<uint32_t> all_lats_collected;
          for (uint32_t i = 0; i < kNumMutatorThreads; i++) { // Collect latencies from all mutator threads
            uint64_t current_lats_idx = lats_idx[i].load(std::memory_order_relaxed);
            auto num_lats = std::min(static_cast<uint64_t>(kLatsWinSize), current_lats_idx);
            // Copy collected latencies into a single vector for sorting
            all_lats_collected.insert(all_lats_collected.end(), &lats[i][0], &lats[i][num_lats]);
          }
 
          if (!all_lats_collected.empty()) {
            std::sort(all_lats_collected.begin(), all_lats_collected.end());
            // Calculate 90th percentile index, adjust for 0-indexed vector
            size_t lat_idx_90 = static_cast<size_t>(all_lats_collected.size() * 90 / 100);
            if (lat_idx_90 >= all_lats_collected.size()) { // Cap at valid index for small sizes
                lat_idx_90 = all_lats_collected.size() - 1;
            }
 
            std::cout << "90 tail lat (us) = " // Latency is now in microseconds
                      << all_lats_collected[lat_idx_90] << std::endl;
          } else {
              std::cout << "90 tail lat (us) = 0 (no latencies)" << std::endl;
          }
          std::exit(0); // Terminate the program cleanly
        }
        prev_us = us;
        prev_sum_reqs = sum_reqs;
      }
      flag.clear(std::memory_order_release); // Release the flag
    }
  }
 
  void bench(AppArray *array) {
    std::vector<std::thread> threads;
    prev_us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::high_resolution_clock::now().time_since_epoch())
                  .count();
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back([&, tid]() { // Lambda captures by reference, `tid` by value
        uint32_t cnt = 0;
        while (true) { // Infinite loop for continuous benchmark
          if (unlikely(cnt++ % kPrintPerIters == 0)) {
            print_perf(); // Call print_perf periodically
          }
 
          // Use thread ID (`tid`) to access thread-specific portions of arrays.
          // This simulates per-core data as was done in Shenango.
          uint32_t req_idx =
              all_zipf_req_indices[tid % kMaxLogicalCPUs][per_core_req_idx[tid].load(std::memory_order_relaxed)];
          // Atomically increment and wrap around for `per_core_req_idx`
          if (unlikely(per_core_req_idx[tid].fetch_add(1, std::memory_order_relaxed) + 1 == kReqSeqLen)) {
            per_core_req_idx[tid].store(0, std::memory_order_relaxed);
          }
 
          auto &req = all_gen_reqs[req_idx];
          Key key;
          std::memcpy(key.data, req.data, kReqLen); // Copy current request data to key
 
          auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
 
          uint32_t array_index = 0;
          {
            // DerefScope concept is removed, direct map access.
            for (uint32_t i = 0; i < kNumKeysPerRequest; i++) {
              append_uint32_to_char_array(i, kLog10NumKeysPerRequest,
                                          key.data + kReqLen); // Modify key for each sub-request
              Value value;
              uint16_t value_len;
              hopscotch_get(key.data, kKeyLen, &value_len, (uint8_t *)value.data);
              array_index += value.num;
            }
          }
          {
            array_index %= kNumArrayEntries;
            // Access `std::vector` directly. Ensure `array_index` is valid.
            if (!array->empty() && array_index < array->size()) {
                const auto &array_entry = array->at(array_index); // Access element by index
                consume_array_entry(array_entry); // Consume the entry (dummy work)
            } else {
                // This case should ideally not be hit with correct constant definitions.
                // Could log an error or assert here.
            }
          }
          auto end_time = std::chrono::high_resolution_clock::now(); // End timing
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
 
          // Record latency for this thread
          uint64_t current_lats_idx = lats_idx[tid].fetch_add(1, std::memory_order_relaxed);
          lats[tid][current_lats_idx % kLatsWinSize] = static_cast<uint32_t>(duration.count());
          
          req_cnts[tid].fetch_add(1, std::memory_order_relaxed); // Increment request count for this thread
        }
      });
    }
 
    // Main thread waits for all mutator threads to finish (which they won't, as it's an infinite loop benchmark).
    // In a real benchmark, there would be a termination condition after some time.
    for (auto &thread : threads) {
      if (thread.joinable()) { // Check if thread is joinable before joining
        thread.join();
      }
    }
  }
 
public:
  // FarMemManager is removed. This function will now orchestrate the local benchmark.
  void do_work() {
    // No FarMemManager to allocate. The `hopscotch_map_std` and `array_std` are members.
    std::cout << "Prepare..." << std::endl;
    prepare_hopscotch(); // Prepare the std::unordered_map and zipf indices.
 
    AppArray array_std; // Instantiate std::vector for the "AppArray"
    prepare_array(&array_std); // Prepare the std::vector
    
    std::cout << "Bench..." << std::endl;
    bench(&array_std); // Pass the std::vector by pointer
  }
 
  // Renamed from `run(netaddr)` to `run()` as network address is no longer relevant.
  void run() {
    // No `madvise` for huge pages in portable standard C++.
    // Removed: BUG_ON(madvise(all_gen_reqs, sizeof(Req) * kNumReqs, MADV_HUGEPAGE) != 0);
 
    // No `FarMemManagerFactory` or `TCPDevice`.
    do_work();
  }
};
} // namespace far_memory
 
// Standard C++ main function
int main(int argc, char *argv[]) {
  // The original `main` function was specific to initializing the Shenango runtime.
  // This is a standard C++ main. No command line arguments are strictly needed
  // for this modified version, as network config is removed.
  // The program will just run the benchmark.
  
  if (argc > 1) {
      std::cout << "Warning: Command line arguments are no longer used by this standard C++ version." << std::endl;
      std::cout << "Usage: " << argv[0] << std::endl;
  }
 
  // Allocate FarMemTest object on the heap using std::unique_ptr
  std::unique_ptr<far_memory::FarMemTest> test_ptr = std::make_unique<far_memory::FarMemTest>();
  test_ptr->run(); // Call run on the unique_ptr

  // far_memory::FarMemTest test;
  // test.run(); // Start the benchmark
 
  return 0; // Standard successful exit code
}