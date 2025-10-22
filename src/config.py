#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件
包含系统的所有配置参数
"""

import os
from pathlib import Path


class Config:
    """系统配置类"""
    
    # ==================== 路径配置 ====================
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent
    
    # 数据目录
    DATA_DIR = PROJECT_ROOT / "复旦中文文本分类语料库"
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"
    
    # 输出目录
    OUTPUT_DIR = PROJECT_ROOT / "output"
    RESULTS_DIR = OUTPUT_DIR / "results"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    LOGS_DIR = OUTPUT_DIR / "logs"
    INDEX_DIR = OUTPUT_DIR / "index"
    
    # 确保输出目录存在
    for dir_path in [OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, INDEX_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== 分词配置 ====================
    # 是否使用停用词
    USE_STOPWORDS = True
    
    # 停用词文件路径（如果有外部停用词文件）
    STOPWORDS_FILE = PROJECT_ROOT / "stopwords.txt"
    
    # 自定义词典路径（如果需要）
    CUSTOM_DICT_FILE = PROJECT_ROOT / "custom_dict.txt"
    
    # 最小词长度（过滤掉长度小于此值的词）
    MIN_WORD_LENGTH = 2
    
    # ==================== BM25参数配置 ====================
    # BM25 k1参数：控制词频饱和度
    # 典型值范围：1.2-2.0，默认1.5
    BM25_K1 = 1.5
    
    # BM25 b参数：控制文档长度归一化程度
    # 范围：0-1，0表示不归一化，1表示完全归一化，默认0.75
    BM25_B = 0.75
    
    # ==================== 索引配置 ====================
    # 索引文件名
    INDEX_FILE = INDEX_DIR / "retrieval_index.pkl"
    
    # 是否使用缓存
    USE_CACHE = True
    
    # 批处理大小（构建索引时）
    BATCH_SIZE = 1000
    
    # ==================== 检索配置 ====================
    # 默认返回的Top-K结果数
    DEFAULT_TOP_K = 10
    
    # 最大返回结果数
    MAX_TOP_K = 100
    
    # 最小相关性分数阈值
    MIN_SCORE_THRESHOLD = 0.0
    
    # ==================== 实验配置 ====================
    # 每个类别加载的最大文档数（None表示加载全部）
    MAX_DOCS_PER_CATEGORY = None  # 改为None以使用完整数据集
    
    # 测试查询列表
    TEST_QUERIES = [
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
        "能源开发电力工业",
    ]
    
    # 参数调优实验配置
    PARAM_TUNING = {
        'k1_values': [0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
        'b_values': [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
    }
    
    # ==================== 评估配置 ====================
    # 评估指标
    EVALUATION_METRICS = [
        'precision@5',
        'precision@10',
        'recall@5',
        'recall@10',
        'map',  # Mean Average Precision
        'ndcg@10',  # Normalized Discounted Cumulative Gain
    ]
    
    # ==================== 可视化配置 ====================
    # 图表DPI
    FIGURE_DPI = 300
    
    # 图表大小
    FIGURE_SIZE = (12, 8)
    
    # 中文字体
    CHINESE_FONT = 'SimHei'
    
    # 颜色方案
    COLOR_SCHEME = 'Set2'
    
    # ==================== 日志配置 ====================
    # 日志级别
    LOG_LEVEL = "INFO"
    
    # 日志格式
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 日志文件
    LOG_FILE = LOGS_DIR / "retrieval_system.log"
    
    # ==================== 编码配置 ====================
    # 支持的编码列表（按优先级排序）
    SUPPORTED_ENCODINGS = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']
    
    # 默认编码
    DEFAULT_ENCODING = 'utf-8'
    
    # ==================== 性能配置 ====================
    # 是否启用多进程
    USE_MULTIPROCESSING = False
    
    # 进程数（None表示使用CPU核心数）
    NUM_PROCESSES = None
    
    # 是否显示进度条
    SHOW_PROGRESS = True
    
    # ==================== 报告配置 ====================
    # 实验报告文件名
    REPORT_FILE = PROJECT_ROOT / "第一次实验报告_稀疏检索算法实现.docx"
    
    # 报告模板文件
    REPORT_TEMPLATE = PROJECT_ROOT / "实验报告模板.docx"
    
    # 报告字体配置
    REPORT_FONTS = {
        'title': {'name': '宋体', 'size': 22, 'bold': True},
        'heading1': {'name': '黑体', 'size': 16, 'bold': False},
        'heading2': {'name': '楷体', 'size': 16, 'bold': True},
        'heading3': {'name': '仿宋', 'size': 16, 'bold': True},
        'body': {'name': '仿宋', 'size': 16, 'bold': False},
    }
    
    # 行间距（磅）
    LINE_SPACING = 28
    
    # 首行缩进（字符数）
    FIRST_LINE_INDENT = 2
    
    @classmethod
    def get_config_dict(cls) -> dict:
        """获取所有配置的字典形式"""
        config_dict = {}
        for key in dir(cls):
            if key.isupper():
                config_dict[key] = getattr(cls, key)
        return config_dict
    
    @classmethod
    def print_config(cls):
        """打印所有配置"""
        print("=" * 80)
        print("系统配置")
        print("=" * 80)
        for key, value in cls.get_config_dict().items():
            print(f"{key}: {value}")
        print("=" * 80)


if __name__ == '__main__':
    Config.print_config()

