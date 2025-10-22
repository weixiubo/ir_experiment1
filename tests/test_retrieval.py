#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索系统测试脚本
测试稀疏检索系统的检索效果
"""

import json
from sparse_retrieval_system import ChineseTokenizer, SparseRetrievalSystem
import time


def test_retrieval_system():
    """测试检索系统"""
    print("=" * 80)
    print("稀疏检索系统测试")
    print("=" * 80)
    
    # 加载检索系统
    print("\n正在加载检索系统...")
    tokenizer = ChineseTokenizer(use_stopwords=True)
    retrieval_system = SparseRetrievalSystem(tokenizer)
    
    try:
        retrieval_system.load_index('retrieval_index.pkl')
    except FileNotFoundError:
        print("索引文件不存在，正在构建...")
        retrieval_system.load_documents('复旦中文文本分类语料库', max_docs_per_category=50)
        retrieval_system.build_index()
        retrieval_system.save_index('retrieval_index.pkl')
    
    # 测试查询
    test_queries = [
        "计算机网络技术",
        "人工智能深度学习",
        "环境保护污染治理",
        "经济发展市场改革",
        "体育运动足球比赛",
        "医疗健康疾病治疗",
        "航天技术卫星发射",
        "农业生产粮食种植",
        "法律法规司法制度",
        "教育改革学校发展",
        "历史文化传统习俗",
        "文学艺术诗歌创作",
        "军事国防武器装备",
        "政治外交国际关系",
        "能源开发电力工业"
    ]
    
    print(f"\n开始测试 {len(test_queries)} 个查询...")
    print("=" * 80)
    
    all_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        print("-" * 80)
        
        # 执行检索
        start_time = time.time()
        results = retrieval_system.search(query, top_k=10)
        elapsed = time.time() - start_time
        
        print(f"检索耗时: {elapsed*1000:.2f} ms")
        print(f"返回结果数: {len(results)}")
        
        if results:
            print(f"\nTop 10 检索结果:")
            query_results = []
            
            for rank, (doc_id, score, doc_info) in enumerate(results, 1):
                category = doc_info['category']
                filename = doc_info['filename']
                text_preview = doc_info['text'][:100].replace('\n', ' ')
                
                print(f"  {rank}. [分数: {score:.4f}] [{category}] {filename}")
                print(f"     预览: {text_preview}...")
                
                query_results.append({
                    'rank': rank,
                    'doc_id': doc_id,
                    'score': score,
                    'category': category,
                    'filename': filename,
                    'text_preview': text_preview
                })
            
            all_results.append({
                'query': query,
                'elapsed_ms': elapsed * 1000,
                'num_results': len(results),
                'results': query_results
            })
        else:
            print("  未找到相关文档")
            all_results.append({
                'query': query,
                'elapsed_ms': elapsed * 1000,
                'num_results': 0,
                'results': []
            })
    
    # 保存测试结果
    with open('retrieval_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print(f"详细结果已保存到: retrieval_test_results.json")
    
    # 统计分析
    print("\n测试统计:")
    avg_time = sum(r['elapsed_ms'] for r in all_results) / len(all_results)
    print(f"  平均检索时间: {avg_time:.2f} ms")
    print(f"  总查询数: {len(all_results)}")
    print(f"  有结果的查询数: {sum(1 for r in all_results if r['num_results'] > 0)}")
    
    return all_results


def interactive_search():
    """交互式检索"""
    print("\n" + "=" * 80)
    print("交互式检索模式")
    print("=" * 80)
    print("输入查询词进行检索，输入 'quit' 退出")
    
    # 加载检索系统
    tokenizer = ChineseTokenizer(use_stopwords=True)
    retrieval_system = SparseRetrievalSystem(tokenizer)
    
    try:
        retrieval_system.load_index('retrieval_index.pkl')
    except FileNotFoundError:
        print("索引文件不存在，正在构建...")
        retrieval_system.load_documents('复旦中文文本分类语料库', max_docs_per_category=50)
        retrieval_system.build_index()
        retrieval_system.save_index('retrieval_index.pkl')
    
    while True:
        print("\n" + "-" * 80)
        query = input("请输入查询: ").strip()
        
        if query.lower() == 'quit':
            print("退出检索系统")
            break
        
        if not query:
            continue
        
        # 执行检索
        start_time = time.time()
        results = retrieval_system.search(query, top_k=10)
        elapsed = time.time() - start_time
        
        print(f"\n检索耗时: {elapsed*1000:.2f} ms")
        print(f"找到 {len(results)} 个相关文档\n")
        
        if results:
            for rank, (doc_id, score, doc_info) in enumerate(results, 1):
                category = doc_info['category']
                filename = doc_info['filename']
                text_preview = doc_info['text'][:150].replace('\n', ' ')
                
                print(f"{rank}. [分数: {score:.4f}] [{category}] {filename}")
                print(f"   {text_preview}...\n")
        else:
            print("未找到相关文档")


def analyze_retrieval_quality(results):
    """分析检索质量"""
    print("\n" + "=" * 80)
    print("检索质量分析")
    print("=" * 80)
    
    # 类别一致性分析
    category_consistency = []
    
    for result in results:
        if result['num_results'] > 0:
            # 检查top结果的类别一致性
            top_categories = [r['category'] for r in result['results'][:5]]
            most_common_category = max(set(top_categories), key=top_categories.count)
            consistency = top_categories.count(most_common_category) / len(top_categories)
            category_consistency.append(consistency)
    
    if category_consistency:
        avg_consistency = sum(category_consistency) / len(category_consistency)
        print(f"\nTop-5 类别一致性: {avg_consistency*100:.2f}%")
        print(f"  (衡量检索结果中前5个文档属于同一类别的比例)")
    
    # 分数分布分析
    all_scores = []
    for result in results:
        if result['num_results'] > 0:
            scores = [r['score'] for r in result['results']]
            all_scores.extend(scores)
    
    if all_scores:
        print(f"\n分数统计:")
        print(f"  最高分: {max(all_scores):.4f}")
        print(f"  最低分: {min(all_scores):.4f}")
        print(f"  平均分: {sum(all_scores)/len(all_scores):.4f}")


if __name__ == '__main__':
    # 运行测试
    results = test_retrieval_system()
    
    # 分析检索质量
    analyze_retrieval_quality(results)
    
    # 交互式检索（可选）
    print("\n是否进入交互式检索模式? (y/n): ", end='')
    choice = input().strip().lower()
    if choice == 'y':
        interactive_search()

