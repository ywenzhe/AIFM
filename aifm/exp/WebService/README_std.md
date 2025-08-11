# WebService Benchmark - C++标准库版本

这是一个基于C++标准库实现的WebService benchmark，用于评估内存上限与运行时间的关系。该版本移除了对AIFM框架的依赖，仅保留了snappy和cryptopp非标准库。

## 功能特性

- **大规模数据集**: 
  - 10GB 哈希表（约2.81亿个键值对）
  - 16GB 数组（约200万个8KB条目）
- **内存限制**: 使用`rlimit`系统调用设置内存上限
- **并发访问**: 多线程并发读写哈希表和数组
- **Zipf分布**: 模拟真实工作负载的访问模式
- **加密压缩**: 每次数组访问都进行AES加密和Snappy压缩

## 文件说明

- `main_std.cpp` - 标准库版本的主程序
- `Makefile_std` - 编译配置文件
- `run_std_simple.sh` - 性能测试脚本
- `run_std.sh` - 使用cgroup的高级测试脚本（需要root权限）

## 编译方法

```bash
make -f Makefile_std
```

### 编译要求
- g++ 7.5+ （支持C++17）
- cryptopp库
- snappy库
- pthread

## 使用方法

### 单次运行
```bash
./main_std <memory_limit_mb>
```

例如：
```bash
./main_std 16384  # 使用16GB内存限制
```

### 批量性能测试
```bash
./run_std_simple.sh
```

该脚本会测试不同内存限制下的性能：
- 8GB - 远小于数据集
- 12GB - 小于数据集  
- 16GB - 接近数据集大小
- 20GB - 稍小于数据集
- 24GB - 接近数据集
- 28GB - 大于数据集
- 32GB - 远大于数据集
- 40GB - 充足内存

## 输出说明

程序输出包括：
- 内存限制设置
- 初始/最终内存使用量
- 数据生成进度
- 运行时间（秒）

示例输出：
```
Memory limit set to 16384 MB
Starting WebService benchmark with 16384 MB memory limit
Initial memory usage: 3932 KB
Prepare...
Initializing random generators...
Initializing encryption...
Generating 8781824 requests and 281018368 KV pairs...
Expected hashmap size: ~10GB, Array size: ~16GB
...
Bench...
runtime_seconds = 45.2341
Final memory usage: 26234567 KB
Memory usage delta: 26230635 KB
```

## 性能分析

该benchmark可以用来分析：
1. **内存压力对性能的影响**: 当可用内存小于数据集大小时，系统会使用swap，导致性能下降
2. **内存访问模式**: Zipf分布模拟热点数据访问
3. **并发性能**: 多线程访问共享数据结构的性能
4. **内存使用效率**: 实际内存使用vs理论数据大小

## 预期结果

- **充足内存** (>28GB): 最佳性能，所有数据在内存中
- **内存紧张** (16-24GB): 部分数据可能被swap，性能下降
- **内存不足** (<16GB): 大量swap操作，性能显著下降

## 与原版AIFM的区别

1. **依赖简化**: 移除了AIFM框架依赖，只保留snappy和cryptopp
2. **数据结构**: 使用标准库的`std::unordered_map`和`std::vector`
3. **内存管理**: 使用系统`rlimit`而不是AIFM的远程内存管理
4. **线程模型**: 使用`std::thread`而不是shenango runtime
5. **保留逻辑**: 保持相同的工作负载和访问模式

## 故障排除

### 编译错误
- 确保安装了cryptopp和snappy开发库
- 检查g++版本是否支持C++17

### 运行时错误  
- 如果内存不足导致程序崩溃，尝试增加内存限制
- 确保系统有足够的swap空间

### 性能问题
- 数据生成阶段需要较长时间（几分钟到几十分钟）
- 可以通过减少数据量来加速测试（修改`kNumKVPairs`和`kNumArrayEntries`）

## 定制化

可以通过修改以下常量来调整测试规模：
- `kNumKVPairs`: 哈希表大小
- `kNumArrayEntries`: 数组大小  
- `kNumMutatorThreads`: 工作线程数
- `kZipfParamS`: Zipf分布参数
