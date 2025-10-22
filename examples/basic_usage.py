#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本使用示例

演示如何使用BM25检索系统进行基本的文档检索。
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sparse_retrieval_system import SparseRetrievalSystem


def main():
    """基本使用示例"""
    
    print("=" * 60)
    print("BM25中文检索系统 - 基本使用示例")
    print("=" * 60)
    
    # 1. 创建检索系统实例
    print("\n1. 创建检索系统...")
    data_dir = Path(__file__).parent.parent / 'data' / '复旦中文文本分类语料库' / 'train'
    
    system = SparseRetrievalSystem(
        data_dir=data_dir,
        k1=1.5,      # BM25参数k1（词频饱和度）
        b=0.75       # BM25参数b（文档长度归一化）
    )
    
    # 2. 构建索引
    print("\n2. 构建索引...")
    system.build_index()
    
    print(f"\n索引构建完成！")
    print(f"  - 文档数量: {len(system.documents)}")
    print(f"  - 词汇量: {len(system.vocabulary)}")
    print(f"  - 平均文档长度: {system.avg_doc_length:.2f}词")
    
    # 3. 执行检索
    print("\n3. 执行检索...")
    
    queries = [
        "计算机网络技术",
        "人工智能深度学习",
        "经济发展市场改革",
        "体育运动足球比赛"
    ]
    
    for query in queries:
        print(f"\n查询: '{query}'")
        print("-" * 60)
        
        results = system.search(query, top_k=5)
        
        if not results:
            print("  未找到相关文档")
            continue
        
        for rank, (doc_id, score) in enumerate(results, 1):
            doc_info = system.documents[doc_id]
            print(f"  {rank}. [{doc_info['category']}] {doc_info['filename']}")
            print(f"     分数: {score:.4f}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

