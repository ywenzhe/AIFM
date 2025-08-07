#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebService基准测试结果分析脚本
分析本地内存容量上限与运行性能的关系
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

def load_results(csv_file):
    """加载基准测试结果"""
    try:
        df = pd.read_csv(csv_file)
        print(f"成功加载 {len(df)} 条测试结果")
        return df
    except FileNotFoundError:
        print(f"错误: 找不到结果文件 {csv_file}")
        return None
    except Exception as e:
        print(f"错误: 加载结果文件失败 - {e}")
        return None

def generate_performance_analysis(df):
    """生成性能分析报告"""
    print("\n=== WebService基准测试性能分析报告 ===")
    print(f"测试配置: 10GB Hashtable + 16GB Array (Synthetic数据集)")
    print(f"测试范围: {df['cache_size_mb'].min()}MB - {df['cache_size_mb'].max()}MB 本地内存")
    print(f"总测试点: {len(df)} 个")
    
    print(f"\n=== 性能指标摘要 ===")
    print(f"最大MOPS: {df['avg_mops'].max():.3f} (缓存大小: {df.loc[df['avg_mops'].idxmax(), 'cache_size_mb']:.0f}MB)")
    print(f"最小MOPS: {df['avg_mops'].min():.3f} (缓存大小: {df.loc[df['avg_mops'].idxmin(), 'cache_size_mb']:.0f}MB)")
    print(f"性能提升: {(df['avg_mops'].max() / df['avg_mops'].min() - 1) * 100:.1f}%")
    
    print(f"\n=== 缺失率分析 ===")
    print(f"Hashtable最低缺失率: {df['hashtable_miss_rate'].min():.4f} (缓存大小: {df.loc[df['hashtable_miss_rate'].idxmin(), 'cache_size_mb']:.0f}MB)")
    print(f"Array最低缺失率: {df['array_miss_rate'].min():.4f} (缓存大小: {df.loc[df['array_miss_rate'].idxmin(), 'cache_size_mb']:.0f}MB)")
    
    print(f"\n=== 延迟分析 ===")
    print(f"最低平均延迟: {df['avg_latency_us'].min():.1f}μs (缓存大小: {df.loc[df['avg_latency_us'].idxmin(), 'cache_size_mb']:.0f}MB)")
    print(f"最低P99延迟: {df['p99_latency_us'].min():.1f}μs (缓存大小: {df.loc[df['p99_latency_us'].idxmin(), 'cache_size_mb']:.0f}MB)")

def create_performance_plots(df, output_dir):
    """创建性能图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 吞吐量 vs 缓存大小
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(df['cache_size_mb'], df['avg_mops'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('本地内存容量 (MB)')
    ax1.set_ylabel('吞吐量 (MOPS)')
    ax1.set_title('本地内存容量 vs 吞吐量')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # 2. 缺失率 vs 缓存大小
    ax2.plot(df['cache_size_mb'], df['hashtable_miss_rate'], 'r-s', 
             linewidth=2, markersize=6, label='Hashtable缺失率')
    ax2.plot(df['cache_size_mb'], df['array_miss_rate'], 'g-^', 
             linewidth=2, markersize=6, label='Array缺失率')
    ax2.set_xlabel('本地内存容量 (MB)')
    ax2.set_ylabel('缺失率')
    ax2.set_title('本地内存容量 vs 缺失率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_vs_memory.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 延迟分析
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(df['cache_size_mb'], df['avg_latency_us'], 'purple', 
             marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('本地内存容量 (MB)')
    ax1.set_ylabel('平均延迟 (μs)')
    ax1.set_title('本地内存容量 vs 平均延迟')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    ax2.plot(df['cache_size_mb'], df['p99_latency_us'], 'orange', 
             marker='s', linewidth=2, markersize=6)
    ax2.set_xlabel('本地内存容量 (MB)')
    ax2.set_ylabel('P99延迟 (μs)')
    ax2.set_title('本地内存容量 vs P99延迟')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latency_vs_memory.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 综合性能分析
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建双Y轴
    ax2 = ax.twinx()
    
    # 吞吐量
    line1 = ax.plot(df['cache_size_mb'], df['avg_mops'], 'b-o', 
                    linewidth=3, markersize=8, label='吞吐量 (MOPS)')
    ax.set_xlabel('本地内存容量 (MB)', fontsize=12)
    ax.set_ylabel('吞吐量 (MOPS)', color='blue', fontsize=12)
    ax.tick_params(axis='y', labelcolor='blue')
    
    # 总缺失率 (hashtable + array的加权平均)
    total_miss_rate = (df['hashtable_miss_rate'] + df['array_miss_rate']) / 2
    line2 = ax2.plot(df['cache_size_mb'], total_miss_rate, 'r-s', 
                     linewidth=3, markersize=8, label='平均缺失率')
    ax2.set_ylabel('平均缺失率', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=12)
    
    ax.set_title('WebService基准测试: 本地内存容量对性能的影响', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"性能图表已保存到 {output_dir}/ 目录")

def calculate_efficiency_metrics(df):
    """计算效率指标"""
    print(f"\n=== 效率指标分析 ===")
    
    # 计算每MB内存的性能提升
    df_sorted = df.sort_values('cache_size_mb')
    df_sorted['mops_per_mb'] = df_sorted['avg_mops'] / df_sorted['cache_size_mb']
    df_sorted['efficiency_ratio'] = df_sorted['avg_mops'] / df_sorted['avg_mops'].iloc[0]
    
    print("内存大小 -> 性能效率 (MOPS/MB) -> 相对性能提升")
    for _, row in df_sorted.iterrows():
        print(f"{row['cache_size_mb']:6.0f}MB -> {row['mops_per_mb']:8.4f} -> {row['efficiency_ratio']:6.2f}x")
    
    # 找到性能拐点 (边际收益递减点)
    df_sorted['mops_diff'] = df_sorted['avg_mops'].diff()
    df_sorted['mops_diff_ratio'] = df_sorted['mops_diff'] / df_sorted['cache_size_mb'].diff()
    
    # 找到性能增长率开始显著下降的点
    if len(df_sorted) > 2:
        inflection_idx = df_sorted['mops_diff_ratio'].idxmax()
        inflection_point = df_sorted.loc[inflection_idx]
        print(f"\n建议的最优内存配置: {inflection_point['cache_size_mb']:.0f}MB")
        print(f"此配置下的性能: {inflection_point['avg_mops']:.3f} MOPS")
        print(f"性价比: {inflection_point['mops_per_mb']:.4f} MOPS/MB")

def save_detailed_report(df, output_file):
    """保存详细报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# WebService基准测试详细报告\n\n")
        f.write("## 测试配置\n")
        f.write("- 数据集: 10GB Hashtable + 16GB Array (Synthetic)\n")
        f.write("- 测试线程: 40\n")
        f.write("- 测试时长: 60秒/配置\n")
        f.write("- Zipf参数: 0.85\n\n")
        
        f.write("## 详细测试结果\n")
        f.write("| 内存大小(MB) | 吞吐量(MOPS) | Hashtable缺失率 | Array缺失率 | 平均延迟(μs) | P99延迟(μs) | 运行时间(s) |\n")
        f.write("|-------------|-------------|----------------|------------|------------|-----------|------------|\n")
        
        for _, row in df.sort_values('cache_size_mb').iterrows():
            f.write(f"| {row['cache_size_mb']:11.0f} | {row['avg_mops']:11.3f} | "
                   f"{row['hashtable_miss_rate']:14.4f} | {row['array_miss_rate']:10.4f} | "
                   f"{row['avg_latency_us']:10.1f} | {row['p99_latency_us']:9.1f} | "
                   f"{row['runtime_sec']:10.1f} |\n")
        
        f.write(f"\n## 性能摘要\n")
        f.write(f"- 最大吞吐量: {df['avg_mops'].max():.3f} MOPS\n")
        f.write(f"- 最小延迟: {df['avg_latency_us'].min():.1f}μs (平均), {df['p99_latency_us'].min():.1f}μs (P99)\n")
        f.write(f"- 性能提升范围: {df['avg_mops'].min():.3f} - {df['avg_mops'].max():.3f} MOPS\n")
        f.write(f"- 测试内存范围: {df['cache_size_mb'].min():.0f}MB - {df['cache_size_mb'].max():.0f}MB\n")

def main():
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = "results/benchmark_results.csv"
    
    # 检查结果文件是否存在
    if not os.path.exists(results_file):
        print(f"错误: 找不到结果文件 {results_file}")
        print("请先运行基准测试: ./run_benchmark.sh")
        return
    
    # 加载结果
    df = load_results(results_file)
    if df is None:
        return
    
    # 数据验证
    if df.empty:
        print("错误: 结果文件为空")
        return
    
    # 生成分析
    generate_performance_analysis(df)
    calculate_efficiency_metrics(df)
    
    # 创建输出目录
    output_dir = "results/analysis"
    
    # 生成图表
    create_performance_plots(df, output_dir)
    
    # 保存详细报告
    save_detailed_report(df, f"{output_dir}/detailed_report.md")
    
    print(f"\n分析完成! 结果已保存到 {output_dir}/ 目录")
    print("生成的文件:")
    print("- performance_vs_memory.png: 性能vs内存图表")
    print("- latency_vs_memory.png: 延迟vs内存图表") 
    print("- comprehensive_analysis.png: 综合分析图表")
    print("- detailed_report.md: 详细报告")

if __name__ == "__main__":
    main()
