#!/bin/bash

source ../../shared.sh

arr_aifm_heap_size=( 3200 6400 12800 25600 )
arr_hashtable_idx_shift=( 25 26 27 28 )

sudo pkill -9 main

for ((i=0;i<${#arr_aifm_heap_size[@]};++i)); do
    cur_heap_size=${arr_aifm_heap_size[i]}
    cur_idx_shift=${arr_hashtable_idx_shift[i]}
    local_mem_size=$(( (1 << ($cur_idx_shift - 20)) * 24 + $cur_heap_size ))

    # --- 新增的提示信息开始 ---
    echo "=================================================="
    echo ">>> 开始处理 Local_mem_size: ${local_mem_size} MB <<<"
    echo "=================================================="
    # --- 新增的提示信息结束 ---

    sed "s/constexpr static uint64_t kCacheSize = .*/constexpr static uint64_t kCacheSize = $cur_heap_size * Region::kSize;/g" main.cpp -i
    sed "s/constexpr static uint32_t kLocalHashTableNumEntriesShift = .*/constexpr static uint32_t kLocalHashTableNumEntriesShift = $cur_idx_shift;/g" main.cpp -i    
    make clean
    make -j
    rerun_local_iokerneld_args simple 1,2,3,4,5,6,7,8,9,11,12,13,14,15
    rerun_mem_server
    run_program ./main | grep "=" 1>log.$local_mem_size 2>&1
done

kill_local_iokerneld
