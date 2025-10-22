#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集统计分析脚本
分析复旦中文文本分类语料库的基本信息
"""

import os
import json
from collections import defaultdict
import chardet


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def read_file_with_encoding(file_path):
    """使用正确的编码读取文件"""
    # 尝试多种编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                if content.strip():  # 确保读取到了内容
                    return content
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # 如果都失败，使用chardet检测
    try:
        detected_encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=detected_encoding) as f:
            return f.read()
    except:
        return ""


def analyze_dataset(data_dir):
    """分析数据集"""
    print("=" * 80)
    print("复旦中文文本分类语料库统计分析")
    print("=" * 80)
    
    # 统计信息
    category_stats = defaultdict(lambda: {
        'count': 0,
        'total_chars': 0,
        'total_lines': 0,
        'files': []
    })
    
    total_files = 0
    total_chars = 0
    total_lines = 0
    
    # 遍历所有类别
    train_dir = os.path.join(data_dir, 'train')
    categories = sorted([d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"\n发现 {len(categories)} 个类别:")
    for cat in categories:
        print(f"  - {cat}")
    
    print("\n正在分析文件...")
    
    # 分析每个类别
    for category in categories:
        cat_dir = os.path.join(train_dir, category)
        files = [f for f in os.listdir(cat_dir) if f.endswith('.txt')]
        
        for filename in files:
            file_path = os.path.join(cat_dir, filename)
            try:
                content = read_file_with_encoding(file_path)
                
                if content:
                    char_count = len(content)
                    line_count = len(content.split('\n'))
                    
                    category_stats[category]['count'] += 1
                    category_stats[category]['total_chars'] += char_count
                    category_stats[category]['total_lines'] += line_count
                    category_stats[category]['files'].append(filename)
                    
                    total_files += 1
                    total_chars += char_count
                    total_lines += line_count
            except Exception as e:
                print(f"  警告: 无法读取文件 {file_path}: {e}")
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    print(f"\n总体统计:")
    print(f"  总文件数: {total_files}")
    print(f"  总字符数: {total_chars:,}")
    print(f"  总行数: {total_lines:,}")
    print(f"  平均每文件字符数: {total_chars/total_files if total_files > 0 else 0:.2f}")
    print(f"  平均每文件行数: {total_lines/total_files if total_files > 0 else 0:.2f}")
    
    print(f"\n各类别统计:")
    print(f"{'类别':<30} {'文件数':>10} {'总字符数':>15} {'平均字符数':>15}")
    print("-" * 80)
    
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        avg_chars = stats['total_chars'] / stats['count'] if stats['count'] > 0 else 0
        print(f"{category:<30} {stats['count']:>10} {stats['total_chars']:>15,} {avg_chars:>15.2f}")
    
    # 保存详细统计到JSON
    output_data = {
        'total_files': total_files,
        'total_chars': total_chars,
        'total_lines': total_lines,
        'categories': {}
    }
    
    for category, stats in category_stats.items():
        output_data['categories'][category] = {
            'count': stats['count'],
            'total_chars': stats['total_chars'],
            'total_lines': stats['total_lines'],
            'avg_chars': stats['total_chars'] / stats['count'] if stats['count'] > 0 else 0,
            'avg_lines': stats['total_lines'] / stats['count'] if stats['count'] > 0 else 0
        }
    
    with open('dataset_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细统计已保存到: dataset_statistics.json")
    
    return output_data


if __name__ == '__main__':
    data_dir = '复旦中文文本分类语料库'
    
    if not os.path.exists(data_dir):
        print(f"错误: 找不到数据目录 {data_dir}")
        exit(1)
    
    stats = analyze_dataset(data_dir)
    print("\n分析完成!")

