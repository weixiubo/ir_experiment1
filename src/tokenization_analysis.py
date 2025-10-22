#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分词统计分析脚本
分析结巴分词的效果和统计信息
"""

import jieba
import jieba.analyse
from collections import Counter, defaultdict
import json
import matplotlib.pyplot as plt
import matplotlib
from sparse_retrieval_system import ChineseTokenizer, SparseRetrievalSystem

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def analyze_tokenization(retrieval_system: SparseRetrievalSystem):
    """分析分词结果"""
    print("=" * 80)
    print("分词统计分析")
    print("=" * 80)
    
    # 统计信息
    total_tokens = 0
    total_unique_tokens = set()
    category_tokens = defaultdict(list)
    category_unique_tokens = defaultdict(set)
    
    # 词频统计
    global_word_freq = Counter()
    category_word_freq = defaultdict(Counter)
    
    # 按类别统计
    for doc_id, tokens in retrieval_system.doc_tokens.items():
        category = retrieval_system.documents[doc_id]['category']
        
        total_tokens += len(tokens)
        total_unique_tokens.update(tokens)
        
        category_tokens[category].extend(tokens)
        category_unique_tokens[category].update(tokens)
        
        global_word_freq.update(tokens)
        category_word_freq[category].update(tokens)
    
    # 输出总体统计
    print(f"\n总体统计:")
    print(f"  总词数: {total_tokens:,}")
    print(f"  唯一词数: {len(total_unique_tokens):,}")
    print(f"  平均每文档词数: {total_tokens / len(retrieval_system.documents):.2f}")
    print(f"  词汇丰富度: {len(total_unique_tokens) / total_tokens * 100:.2f}%")
    
    # 高频词
    print(f"\n全局高频词 (Top 30):")
    for word, freq in global_word_freq.most_common(30):
        print(f"  {word}: {freq}")
    
    # 按类别统计
    print(f"\n各类别分词统计:")
    print(f"{'类别':<30} {'文档数':>10} {'总词数':>12} {'唯一词数':>12} {'平均词数':>12}")
    print("-" * 80)
    
    category_stats = {}
    for category in sorted(category_tokens.keys()):
        doc_count = sum(1 for doc in retrieval_system.documents.values() 
                       if doc['category'] == category)
        total = len(category_tokens[category])
        unique = len(category_unique_tokens[category])
        avg = total / doc_count if doc_count > 0 else 0
        
        print(f"{category:<30} {doc_count:>10} {total:>12,} {unique:>12,} {avg:>12.2f}")
        
        category_stats[category] = {
            'doc_count': doc_count,
            'total_tokens': total,
            'unique_tokens': unique,
            'avg_tokens': avg,
            'top_words': [{'word': w, 'freq': f} for w, f in category_word_freq[category].most_common(20)]
        }
    
    # 保存统计结果
    output_data = {
        'total_tokens': total_tokens,
        'unique_tokens': len(total_unique_tokens),
        'avg_tokens_per_doc': total_tokens / len(retrieval_system.documents),
        'vocabulary_richness': len(total_unique_tokens) / total_tokens,
        'global_top_words': [{'word': w, 'freq': f} for w, f in global_word_freq.most_common(100)],
        'category_stats': category_stats
    }
    
    with open('tokenization_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细统计已保存到: tokenization_statistics.json")
    
    # 生成可视化
    generate_visualizations(global_word_freq, category_word_freq, category_stats)
    
    return output_data


def generate_visualizations(global_word_freq, category_word_freq, category_stats):
    """生成可视化图表"""
    print("\n正在生成可视化图表...")
    
    # 1. 全局高频词柱状图
    plt.figure(figsize=(12, 6))
    top_words = global_word_freq.most_common(20)
    words = [w for w, _ in top_words]
    freqs = [f for _, f in top_words]
    
    plt.bar(range(len(words)), freqs)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.xlabel('词语')
    plt.ylabel('频次')
    plt.title('全局高频词 Top 20')
    plt.tight_layout()
    plt.savefig('global_top_words.png', dpi=300, bbox_inches='tight')
    print("  已保存: global_top_words.png")
    plt.close()
    
    # 2. 各类别词汇量对比
    plt.figure(figsize=(14, 6))
    categories = sorted(category_stats.keys())
    unique_counts = [category_stats[cat]['unique_tokens'] for cat in categories]
    
    plt.bar(range(len(categories)), unique_counts)
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
    plt.xlabel('类别')
    plt.ylabel('唯一词数')
    plt.title('各类别唯一词数对比')
    plt.tight_layout()
    plt.savefig('category_vocabulary_size.png', dpi=300, bbox_inches='tight')
    print("  已保存: category_vocabulary_size.png")
    plt.close()
    
    # 3. 各类别平均文档长度
    plt.figure(figsize=(14, 6))
    avg_lengths = [category_stats[cat]['avg_tokens'] for cat in categories]
    
    plt.bar(range(len(categories)), avg_lengths)
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
    plt.xlabel('类别')
    plt.ylabel('平均词数')
    plt.title('各类别平均文档长度（词数）')
    plt.tight_layout()
    plt.savefig('category_avg_length.png', dpi=300, bbox_inches='tight')
    print("  已保存: category_avg_length.png")
    plt.close()


def demonstrate_tokenization():
    """演示分词效果"""
    print("\n" + "=" * 80)
    print("分词效果演示")
    print("=" * 80)
    
    # 示例文本
    examples = [
        "中国科学院计算技术研究所是中国计算机科学研究的重要基地",
        "人工智能技术在医疗诊断领域的应用越来越广泛",
        "北京大学和清华大学是中国最著名的高等学府",
        "深度学习算法在图像识别任务中取得了突破性进展"
    ]
    
    tokenizer = ChineseTokenizer(use_stopwords=False)
    tokenizer_with_stopwords = ChineseTokenizer(use_stopwords=True)
    
    for i, text in enumerate(examples, 1):
        print(f"\n示例 {i}: {text}")
        
        # 不使用停用词
        tokens = tokenizer.tokenize(text)
        print(f"  分词结果（无停用词过滤）: {' / '.join(tokens)}")
        
        # 使用停用词
        tokens_filtered = tokenizer_with_stopwords.tokenize(text)
        print(f"  分词结果（停用词过滤）: {' / '.join(tokens_filtered)}")
        
        # 关键词提取
        keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
        print(f"  关键词提取: {', '.join([f'{w}({s:.3f})' for w, s in keywords])}")


if __name__ == '__main__':
    # 演示分词效果
    demonstrate_tokenization()
    
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
    
    # 分析分词结果
    stats = analyze_tokenization(retrieval_system)
    
    print("\n分析完成!")

