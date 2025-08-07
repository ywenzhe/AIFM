#!/bin/bash

echo "WebService基准测试程序 - 快速验证"
echo "================================="

# 检查必要文件
echo "1. 检查文件完整性..."
required_files=(
    "benchmark_main.cpp"
    "Makefile.benchmark" 
    "run_benchmark.sh"
    "analyze_results.py"
    "README_benchmark.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (缺失)"
        exit 1
    fi
done

# 检查依赖路径
echo "2. 检查依赖路径..."
if [ -d "../../inc" ]; then
    echo "  ✓ AIFM头文件路径"
else
    echo "  ✗ AIFM头文件路径 (../../inc)"
    exit 1
fi

if [ -d "../../../shenango" ]; then
    echo "  ✓ Shenango路径"
else
    echo "  ✗ Shenango路径 (../../../shenango)"
    exit 1
fi

# 尝试编译
echo "3. 编译测试..."
make -f Makefile.benchmark clean > /dev/null 2>&1
if make -f Makefile.benchmark -j > compile.log 2>&1; then
    echo "  ✓ 编译成功"
    rm -f compile.log
else
    echo "  ✗ 编译失败，查看错误信息:"
    cat compile.log
    exit 1
fi

# 检查可执行文件
if [ -f "benchmark_main" ]; then
    echo "  ✓ 可执行文件已生成"
    file_size=$(stat -f%z benchmark_main 2>/dev/null || stat -c%s benchmark_main 2>/dev/null)
    echo "    文件大小: $((file_size / 1024))KB"
else
    echo "  ✗ 可执行文件未生成"
    exit 1
fi

# 检查脚本权限
echo "4. 检查脚本权限..."
if [ -x "run_benchmark.sh" ]; then
    echo "  ✓ run_benchmark.sh 可执行"
else
    echo "  ! 设置 run_benchmark.sh 执行权限"
    chmod +x run_benchmark.sh
fi

if [ -x "analyze_results.py" ]; then
    echo "  ✓ analyze_results.py 可执行"
else
    echo "  ! 设置 analyze_results.py 执行权限"
    chmod +x analyze_results.py
fi

# 检查Python依赖
echo "5. 检查Python环境..."
if command -v python3 &> /dev/null; then
    echo "  ✓ Python3 可用"
    python3 -c "import pandas, matplotlib, numpy" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✓ Python依赖库可用 (pandas, matplotlib, numpy)"
    else
        echo "  ! Python依赖库缺失，请安装: pip3 install pandas matplotlib numpy"
    fi
else
    echo "  ✗ Python3 未安装"
fi

# 显示使用说明
echo ""
echo "验证完成！使用说明:"
echo "==================="
echo ""
echo "运行完整基准测试:"
echo "  ./run_benchmark.sh"
echo ""
echo "运行单个测试 (例: 1.1GB缓存):"
echo "  source ../../shared.sh"
echo "  rerun_local_iokerneld_args simple 1,2,3,4,5,6,7,8,9,11,12,13,14,15"
echo "  rerun_mem_server"
echo "  ./benchmark_main client.config 192.168.1.100:9999 563"
echo "  kill_local_iokerneld"
echo ""
echo "分析结果:"
echo "  python3 analyze_results.py"
echo ""
echo "查看详细说明:"
echo "  cat README_benchmark.md"

echo ""
echo "程序已准备就绪！"
