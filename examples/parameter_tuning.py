#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数调优示例

演示如何调优BM25的k1和b参数以获得最佳检索效果。
"""

import sys
from pathlib import Path
import time

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sparse_retrieval_system import SparseRetrievalSystem


def evaluate_parameters(system, test_queries, k1, b):
    """
    评估给定参数的检索效果
    
    Args:
        system: 检索系统实例
        test_queries: 测试查询列表
        k1: BM25参数k1
        b: BM25参数b
    
    Returns:
        平均检索时间（毫秒）
    """
    system.k1 = k1
    system.b = b
    
    total_time = 0
    for query in test_queries:
        start_time = time.time()
        system.search(query, top_k=10)
        total_time += (time.time() - start_time) * 1000
    
    avg_time = total_time / len(test_queries)
    return avg_time


def main():
    """参数调优示例"""
    
    print("=" * 60)
    print("BM25参数调优示例")
    print("=" * 60)
    
    # 1. 创建检索系统
    print("\n1. 创建检索系统并构建索引...")
    data_dir = Path(__file__).parent.parent / 'data' / '复旦中文文本分类语料库' / 'train'
    
    system = SparseRetrievalSystem(data_dir=data_dir)
    system.build_index()
    
    # 2. 准备测试查询
    test_queries = [
        "计算机网络技术",
        "人工智能深度学习",
        "经济发展市场改革",
        "体育运动足球比赛",
        "医疗健康保险",
        "环境保护污染",
        "教育改革政策",
        "军事国防安全"
    ]
    
    # 3. 网格搜索
    print("\n2. 开始参数网格搜索...")
    print("\n参数组合测试结果：")
    print("-" * 60)
    print(f"{'k1':<6} {'b':<6} {'平均检索时间(ms)':<20}")
    print("-" * 60)
    
    k1_values = [1.2, 1.5, 1.8]
    b_values = [0.5, 0.75, 0.9]
    
    best_params = None
    best_time = float('inf')
    
    for k1 in k1_values:
        for b in b_values:
            avg_time = evaluate_parameters(system, test_queries, k1, b)
            print(f"{k1:<6.1f} {b:<6.2f} {avg_time:<20.2f}")
            
            if avg_time < best_time:
                best_time = avg_time
                best_params = (k1, b)
    
    print("-" * 60)
    print(f"\n最佳参数组合: k1={best_params[0]}, b={best_params[1]}")
    print(f"平均检索时间: {best_time:.2f}ms")
    
    # 4. 使用最佳参数进行检索
    print("\n3. 使用最佳参数进行检索示例...")
    system.k1, system.b = best_params
    
    query = "人工智能深度学习"
    print(f"\n查询: '{query}'")
    print("-" * 60)
    
    results = system.search(query, top_k=5)
    for rank, (doc_id, score) in enumerate(results, 1):
        doc_info = system.documents[doc_id]
        print(f"{rank}. [{doc_info['category']}] {doc_info['filename']}")
        print(f"   分数: {score:.4f}")
    
    print("\n" + "=" * 60)
    print("参数调优完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

