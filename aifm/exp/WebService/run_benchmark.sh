#!/bin/bash

source ../../shared.sh

# 基准测试配置
# 缓存大小配置 (以Region为单位，每个Region约2MB)
# 测试不同的本地内存容量上限
declare -a cache_sizes=(
    563    # ~1.1GB
    1126   # ~2.2GB  
    2252   # ~4.4GB
    4504   # ~8.8GB
    6756   # ~13.2GB
    9008   # ~17.6GB
    11260  # ~22GB
    13512  # ~26.4GB
    15764  # ~30.8GB
    18016  # ~35.2GB
    20268  # ~39.6GB
)

# 创建结果文件夹
mkdir -p results
cd results

# 创建CSV文件头
echo "cache_size_mb,avg_mops,hashtable_miss_rate,array_miss_rate,runtime_sec,total_requests,avg_latency_us,p99_latency_us" > benchmark_results.csv

cd ..

echo "开始WebService基准测试..."
echo "数据集配置: 10GB Hashtable + 16GB Array"
echo "测试不同本地内存容量上限的性能表现"

# 停止之前可能运行的程序
sudo pkill -9 benchmark_main

# 编译基准测试程序
echo "编译基准测试程序..."
make -f Makefile.benchmark clean
make -f Makefile.benchmark -j

if [ ! -f benchmark_main ]; then
    echo "编译失败，请检查错误信息"
    exit 1
fi

# 遍历不同的缓存大小进行测试
for cache_size in "${cache_sizes[@]}"; do
    echo "========================================="
    echo "测试缓存大小: ${cache_size} regions (~$((cache_size * 2))MB)"
    echo "========================================="
    
    # 启动本地IO内核和内存服务器
    rerun_local_iokerneld_args simple 1,2,3,4,5,6,7,8,9,11,12,13,14,15
    sleep 2
    rerun_mem_server
    sleep 2
    
    # 运行基准测试
    echo "开始基准测试..."
    timeout 300 run_program ./benchmark_main client.config 192.168.1.100:9999 ${cache_size} 2>&1 | tee results/log.${cache_size}
    
    # 检查是否成功完成
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "缓存大小 ${cache_size} 的测试完成"
    else
        echo "缓存大小 ${cache_size} 的测试失败或超时"
    fi
    
    # 停止服务器
    kill_local_iokerneld
    sleep 2
    
    echo "等待系统稳定..."
    sleep 5
done

echo "========================================="
echo "所有基准测试完成！"
echo "结果已保存到 results/benchmark_results.csv"
echo "详细日志保存在 results/log.* 文件中"
echo "========================================="

# 生成简单的结果摘要
if [ -f results/benchmark_results.csv ]; then
    echo ""
    echo "结果摘要:"
    echo "缓存大小(MB), 平均MOPS, Hashtable缺失率, Array缺失率, 运行时间(秒)"
    tail -n +2 results/benchmark_results.csv | awk -F',' '{printf "%d, %.3f, %.3f, %.3f, %.1f\n", $1, $2, $3, $4, $5}'
fi
