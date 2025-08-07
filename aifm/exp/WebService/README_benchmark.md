# WebService基准测试程序

本程序基于AIFM论文中的WebService前端应用程序，用于评估本地内存容量上限与运行性能的关系。

## 测试配置

- **数据集**: Synthetic数据集
  - Hashtable: 10GB (约6.4亿键值对)
  - Array: 16GB (约200万个8KB条目)
- **测试线程**: 40个mutator线程
- **访问模式**: Zipf分布 (s=0.85)
- **测试时长**: 每个配置60秒
- **远程内存**: 20GB

## 文件说明

- `benchmark_main.cpp`: 主要的基准测试程序
- `Makefile.benchmark`: 编译配置文件
- `run_benchmark.sh`: 自动化测试脚本
- `analyze_results.py`: 结果分析脚本
- `README_benchmark.md`: 本说明文件

## 快速开始

### 1. 编译程序

```bash
make -f Makefile.benchmark clean
make -f Makefile.benchmark -j
```

### 2. 运行基准测试

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

### 3. 分析结果

```bash
python3 analyze_results.py
```

## 手动运行

如果需要手动运行特定配置的测试：

```bash
# 启动服务
source ../../shared.sh
rerun_local_iokerneld_args simple 1,2,3,4,5,6,7,8,9,11,12,13,14,15
rerun_mem_server

# 运行测试 (缓存大小以Region为单位，每个Region约2MB)
./benchmark_main client.config 192.168.1.100:9999 563

# 清理
kill_local_iokerneld
```

## 测试参数说明

### 缓存大小配置

程序通过命令行参数指定本地缓存大小（以Region为单位）：

- 563 regions ≈ 1.1GB
- 1126 regions ≈ 2.2GB
- 2252 regions ≈ 4.4GB
- 4504 regions ≈ 8.8GB
- ...
- 20268 regions ≈ 39.6GB

### 性能指标

程序输出以下性能指标：

- **MOPS**: 每秒百万次操作
- **Hashtable缺失率**: 远程访问的比例
- **Array缺失率**: 远程访问的比例
- **延迟**: 平均延迟和P99延迟
- **总请求数**: 测试期间处理的请求总数

## 结果文件

测试完成后，结果保存在以下文件中：

- `results/benchmark_results.csv`: CSV格式的原始结果
- `results/log.*`: 每个配置的详细日志
- `results/analysis/`: 分析结果和图表

## 自定义配置

### 修改数据集大小

在 `benchmark_main.cpp` 中修改以下常量：

```cpp
// Hashtable大小 (10GB)
constexpr static uint32_t kRemoteHashTableNumEntriesShift = 30;
constexpr static uint64_t kRemoteHashTableSlabSize = (10ULL << 30) * 1.05;

// Array大小 (16GB)
constexpr static uint32_t kNumArrayEntries = (16ULL << 30) / 8192;
```

### 修改测试时长

```cpp
constexpr static uint32_t kBenchmarkDurationSec = 60; // 秒
```

### 修改线程数

```cpp
constexpr static uint32_t kNumMutatorThreads = 40;
```

## 故障排除

### 编译错误

1. 确保已安装所需依赖：
   - CryptoPP库
   - Snappy压缩库
   - Shenango运行时

2. 检查路径配置：
   ```bash
   ls ../../inc/  # 应该包含AIFM头文件
   ls ../../../shenango/  # 应该包含Shenango
   ```

### 运行时错误

1. **内存不足**: 减少数据集大小或增加系统内存
2. **网络连接失败**: 检查IP地址和端口配置
3. **权限错误**: 确保有sudo权限启动IOKernel

### 性能异常

1. **吞吐量过低**: 检查CPU和内存使用情况
2. **缺失率异常**: 验证缓存大小配置
3. **延迟过高**: 检查网络延迟和系统负载

## 扩展功能

### 添加新的性能指标

在 `BenchmarkResults` 结构体中添加新字段，并在 `calculate_results()` 中计算。

### 修改访问模式

替换Zipf分布为其他分布（如均匀分布、正态分布等）。

### 支持多种数据集

扩展程序以支持真实数据集或其他Synthetic数据集。

## 参考

- AIFM: High-Performance, Application-Integrated Far Memory 论文
- Shenango运行时文档
- AIFM项目主页
