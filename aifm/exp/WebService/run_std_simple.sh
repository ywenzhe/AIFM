#!/bin/bash

# WebService Benchmark 标准库版本 - 内存限制与性能关系测试
# 使用rlimit进行内存限制，无需root权限

echo "=================================================="
echo "WebService Benchmark - 标准库版本"
echo "内存限制与性能关系测试 (使用rlimit)"
echo "=================================================="

# 检查程序是否存在
if [ ! -f "./main_std" ]; then
    echo "错误: main_std 不存在，请先编译程序"
    echo "运行: make -f Makefile_std"
    exit 1
fi

# 内存限制数组（MB）- 针对26GB数据集进行测试
memory_limits=(
    8192   # 8GB - 远小于数据集
    12288  # 12GB - 小于数据集
    16384  # 16GB - 接近数据集大小
    20480  # 20GB - 稍小于数据集
    24576  # 24GB - 接近数据集
    28672  # 28GB - 大于数据集
    32768  # 32GB - 远大于数据集
    40960  # 40GB - 充足内存
)

# 结果文件
result_file="benchmark_results_$(date +%Y%m%d_%H%M%S).csv"
echo "Memory_Limit_MB,Runtime_Seconds,Exit_Code,Log_File" > "$result_file"

echo "开始测试，结果将保存到: $result_file"
echo ""

# 对每个内存限制进行测试
for memory_limit in "${memory_limits[@]}"; do
    echo "=================================================="
    echo "测试内存限制: ${memory_limit}MB"
    echo "=================================================="
    
    log_file="log_std_${memory_limit}MB.txt"
    
    # 记录开始时间
    start_time=$(date +%s.%N)
    
    # 运行测试
    echo "运行程序: ./main_std $memory_limit"
    echo "日志文件: $log_file"
    
    ./main_std "$memory_limit" > "$log_file" 2>&1
    exit_code=$?
    
    # 记录结束时间
    end_time=$(date +%s.%N)
    runtime=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || python3 -c "print($end_time - $start_time)")
    
    # 记录结果
    echo "$memory_limit,$runtime,$exit_code,$log_file" >> "$result_file"
    
    echo "运行时间: ${runtime}秒"
    echo "退出码: $exit_code"
    
    # 显示运行时输出的关键信息
    if [ -f "$log_file" ]; then
        echo "关键输出:"
        grep -E "(runtime_seconds|Memory limit|error|Error)" "$log_file" | head -5
    fi
    
    echo ""
    
    # 短暂休息
    sleep 1
done

echo "=================================================="
echo "所有测试完成！"
echo "结果保存在: $result_file"
echo "=================================================="

# 显示结果摘要
echo ""
echo "测试结果摘要:"
echo "Memory_Limit_MB,Runtime_Seconds,Exit_Code,Log_File"
cat "$result_file" | tail -n +2

# 生成性能分析报告
echo ""
echo "=================================================="
echo "性能分析报告"
echo "=================================================="

# 提取成功运行的结果
successful_runs=$(awk -F',' 'NR>1 && $3==0 {print $1 "," $2}' "$result_file")

if [ -n "$successful_runs" ]; then
    echo "成功运行的测试:"
    echo "Memory_Limit_MB,Runtime_Seconds"
    echo "$successful_runs"
    
    # 找出最快的运行
    fastest=$(echo "$successful_runs" | sort -t',' -k2 -n | head -1)
    if [ -n "$fastest" ]; then
        fastest_memory=$(echo "$fastest" | cut -d',' -f1)
        fastest_time=$(echo "$fastest" | cut -d',' -f2)
        echo ""
        echo "最佳性能: ${fastest_memory}MB 内存限制，运行时间 ${fastest_time}秒"
    fi
else
    echo "没有成功运行的测试"
fi

echo ""
echo "详细日志文件:"
ls -la log_std_*.txt 2>/dev/null || echo "没有找到日志文件"
