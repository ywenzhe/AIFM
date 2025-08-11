#!/bin/bash

# WebService Benchmark 标准库版本 - 内存限制与性能关系测试

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then
    echo "请以root权限运行此脚本以设置cgroup内存限制"
    echo "使用: sudo $0"
    exit 1
fi

# 设置cgroup名称
CGROUP_NAME="webservice_benchmark"
CGROUP_PATH="/sys/fs/cgroup/memory/$CGROUP_NAME"

# 创建cgroup
setup_cgroup() {
    local memory_limit_mb=$1
    
    echo "设置cgroup内存限制: ${memory_limit_mb}MB"
    
    # 删除已存在的cgroup（如果存在）
    if [ -d "$CGROUP_PATH" ]; then
        echo "删除已存在的cgroup..."
        rmdir "$CGROUP_PATH" 2>/dev/null || true
    fi
    
    # 创建新的cgroup
    mkdir -p "$CGROUP_PATH"
    
    # 设置内存限制（MB转换为字节）
    local memory_limit_bytes=$((memory_limit_mb * 1024 * 1024))
    echo "$memory_limit_bytes" > "$CGROUP_PATH/memory.limit_in_bytes"
    
    # 禁用swap（可选）
    echo 0 > "$CGROUP_PATH/memory.swappiness" 2>/dev/null || true
    
    echo "Cgroup创建完成，内存限制: ${memory_limit_mb}MB"
}

# 清理cgroup
cleanup_cgroup() {
    if [ -d "$CGROUP_PATH" ]; then
        echo "清理cgroup..."
        # 移除所有进程
        if [ -f "$CGROUP_PATH/cgroup.procs" ]; then
            cat "$CGROUP_PATH/cgroup.procs" | while read pid; do
                if [ -n "$pid" ]; then
                    kill -TERM "$pid" 2>/dev/null || true
                fi
            done
        fi
        sleep 1
        rmdir "$CGROUP_PATH" 2>/dev/null || true
    fi
}

# 在cgroup中运行程序
run_in_cgroup() {
    local memory_limit_mb=$1
    local log_file="log_std_${memory_limit_mb}MB"
    
    echo "在cgroup中运行程序..."
    echo "内存限制: ${memory_limit_mb}MB"
    echo "日志文件: $log_file"
    
    # 启动程序并将其PID添加到cgroup
    ./main_std "$memory_limit_mb" > "$log_file" 2>&1 &
    local pid=$!
    
    # 将进程添加到cgroup
    echo "$pid" > "$CGROUP_PATH/cgroup.procs"
    
    echo "程序PID: $pid，已添加到cgroup"
    
    # 等待程序完成
    wait "$pid"
    local exit_code=$?
    
    echo "程序完成，退出码: $exit_code"
    
    # 显示内存使用统计
    if [ -f "$CGROUP_PATH/memory.max_usage_in_bytes" ]; then
        local max_usage=$(cat "$CGROUP_PATH/memory.max_usage_in_bytes")
        local max_usage_mb=$((max_usage / 1024 / 1024))
        echo "最大内存使用量: ${max_usage_mb}MB"
        echo "内存限制: ${memory_limit_mb}MB"
        echo "内存使用率: $(echo "scale=2; $max_usage_mb * 100 / $memory_limit_mb" | bc)%"
    fi
    
    return $exit_code
}

# 主函数
main() {
    echo "=================================================="
    echo "WebService Benchmark - 标准库版本"
    echo "内存限制与性能关系测试"
    echo "=================================================="
    
    # 检查程序是否存在
    if [ ! -f "./main_std" ]; then
        echo "错误: main_std 不存在，请先编译程序"
        echo "运行: make -f Makefile_std"
        exit 1
    fi
    
    # 检查依赖
    command -v bc >/dev/null 2>&1 || { echo "错误: 需要安装bc计算器"; exit 1; }
    
    # 内存限制数组（MB）
    memory_limits=(
        512    # 512MB
        1024   # 1GB
        2048   # 2GB
        4096   # 4GB
        8192   # 8GB
        12288  # 12GB
        16384  # 16GB
        20480  # 20GB
    )
    
    # 结果文件
    result_file="benchmark_results_$(date +%Y%m%d_%H%M%S).csv"
    echo "Memory_Limit_MB,Runtime_Seconds,Max_Memory_Usage_MB,Memory_Usage_Percent,Exit_Code" > "$result_file"
    
    # 设置清理陷阱
    trap cleanup_cgroup EXIT
    
    echo "开始测试，结果将保存到: $result_file"
    echo ""
    
    # 对每个内存限制进行测试
    for memory_limit in "${memory_limits[@]}"; do
        echo "=================================================="
        echo "测试内存限制: ${memory_limit}MB"
        echo "=================================================="
        
        # 设置cgroup
        setup_cgroup "$memory_limit"
        
        # 记录开始时间
        start_time=$(date +%s.%N)
        
        # 运行测试
        run_in_cgroup "$memory_limit"
        exit_code=$?
        
        # 记录结束时间
        end_time=$(date +%s.%N)
        runtime=$(echo "$end_time - $start_time" | bc)
        
        # 获取内存使用统计
        max_usage_mb=0
        usage_percent=0
        if [ -f "$CGROUP_PATH/memory.max_usage_in_bytes" ]; then
            local max_usage=$(cat "$CGROUP_PATH/memory.max_usage_in_bytes")
            max_usage_mb=$((max_usage / 1024 / 1024))
            usage_percent=$(echo "scale=2; $max_usage_mb * 100 / $memory_limit" | bc)
        fi
        
        # 记录结果
        echo "$memory_limit,$runtime,$max_usage_mb,$usage_percent,$exit_code" >> "$result_file"
        
        echo "运行时间: ${runtime}秒"
        echo "最大内存使用: ${max_usage_mb}MB (${usage_percent}%)"
        echo "退出码: $exit_code"
        echo ""
        
        # 清理当前cgroup
        cleanup_cgroup
        
        # 短暂休息
        sleep 2
    done
    
    echo "=================================================="
    echo "所有测试完成！"
    echo "结果保存在: $result_file"
    echo "=================================================="
    
    # 显示结果摘要
    echo ""
    echo "测试结果摘要:"
    echo "Memory_Limit_MB,Runtime_Seconds,Max_Memory_Usage_MB,Memory_Usage_Percent,Exit_Code"
    cat "$result_file" | tail -n +2
}

# 运行主函数
main "$@"
