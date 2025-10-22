#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稀疏检索系统实现

本模块实现了一个基于BM25算法的中文文本稀疏检索系统。
主要功能包括：
    - 中文分词（基于结巴分词）
    - 倒排索引构建
    - BM25相关性评分
    - 文档检索和排序

作者：华中师范大学计算机科学系
日期：2025年10月
"""

import os
import json
import math
import pickle
import chardet
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set, Any
from tqdm import tqdm

import jieba
import jieba.analyse

from config import Config
from logger import get_logger


# 获取日志记录器
logger = get_logger(__name__)


class ChineseTokenizer:
    """
    中文分词器

    使用结巴分词进行中文文本分词，支持停用词过滤和自定义词典。

    Attributes:
        use_stopwords (bool): 是否使用停用词过滤
        stopwords (Set[str]): 停用词集合
        min_word_length (int): 最小词长度

    Examples:
        >>> tokenizer = ChineseTokenizer(use_stopwords=True)
        >>> tokens = tokenizer.tokenize("这是一个测试文本")
        >>> print(tokens)
        ['测试', '文本']
    """

    def __init__(self,
                 use_stopwords: bool = True,
                 stopwords_file: Optional[Path] = None,
                 custom_dict_file: Optional[Path] = None,
                 min_word_length: int = 2):
        """
        初始化中文分词器

        Args:
            use_stopwords: 是否使用停用词过滤
            stopwords_file: 停用词文件路径，如果为None则使用内置停用词
            custom_dict_file: 自定义词典文件路径
            min_word_length: 最小词长度，小于此长度的词将被过滤
        """
        self.use_stopwords = use_stopwords
        self.min_word_length = min_word_length
        self.stopwords = self._load_stopwords(stopwords_file) if use_stopwords else set()

        # 加载自定义词典
        if custom_dict_file and custom_dict_file.exists():
            jieba.load_userdict(str(custom_dict_file))
            logger.info(f"已加载自定义词典: {custom_dict_file}")

        logger.info(f"中文分词器初始化完成 (停用词: {len(self.stopwords)}个, 最小词长: {min_word_length})")

    def _load_stopwords(self, stopwords_file: Optional[Path] = None) -> Set[str]:
        """
        加载停用词

        Args:
            stopwords_file: 停用词文件路径

        Returns:
            停用词集合
        """
        # 如果提供了外部停用词文件
        if stopwords_file and stopwords_file.exists():
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    stopwords = set(line.strip() for line in f if line.strip())
                logger.info(f"从文件加载停用词: {len(stopwords)}个")
                return stopwords
            except Exception as e:
                logger.warning(f"加载停用词文件失败: {e}，使用内置停用词")

        # 使用内置停用词
        stopwords = {
            # 代词
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
            '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '为', '与',
            '他', '她', '它', '们', '这个', '那个', '这些', '那些', '什么', '怎么', '为什么', '哪里',
            # 连词
            '及', '等', '但', '而', '或', '因为', '所以', '如果', '虽然', '但是', '然而', '因此',
            # 介词
            '之', '于', '以', '对', '从', '由', '向', '把', '被', '让', '给', '将', '使', '用',
            # 量词和时间词
            '年', '月', '日', '时', '分', '秒', '个', '只', '件', '条', '张', '次', '回', '下', '中',
            # 其他常用虚词
            '啊', '呀', '吗', '呢', '吧', '哦', '哈', '嗯', '哎', '唉',
        }
        return stopwords

    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词

        Args:
            text: 待分词的文本

        Returns:
            分词结果列表

        Examples:
            >>> tokenizer = ChineseTokenizer()
            >>> tokens = tokenizer.tokenize("中国人工智能技术发展迅速")
            >>> print(tokens)
            ['中国', '人工智能', '技术', '发展', '迅速']
        """
        if not text or not text.strip():
            return []

        # 使用结巴分词
        words = jieba.cut(text.strip())

        # 过滤停用词和短词
        filtered_words = []
        for word in words:
            word = word.strip()
            # 过滤空词和过短的词
            if len(word) < self.min_word_length:
                continue
            # 过滤停用词
            if self.use_stopwords and word in self.stopwords:
                continue
            filtered_words.append(word)

        return filtered_words

    def extract_keywords(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        提取文本关键词

        Args:
            text: 待提取关键词的文本
            top_k: 返回的关键词数量

        Returns:
            关键词及其权重的列表，格式为 [(词, 权重), ...]
        """
        keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)
        return keywords


class SparseRetrievalSystem:
    """
    稀疏检索系统

    基于BM25算法的文本检索系统，支持中文文档的索引构建和检索。

    BM25算法公式：
        Score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1-b+b×|D|/avgdl))

    其中：
        - D: 文档
        - Q: 查询
        - qi: 查询中的第i个词
        - f(qi,D): 词qi在文档D中的频率
        - |D|: 文档D的长度
        - avgdl: 平均文档长度
        - k1, b: 调节参数

    Attributes:
        tokenizer (ChineseTokenizer): 分词器
        documents (Dict): 文档集合
        inverted_index (Dict): 倒排索引
        doc_lengths (Dict): 文档长度索引
        vocabulary (Set): 词汇表
        k1 (float): BM25参数k1
        b (float): BM25参数b

    Examples:
        >>> tokenizer = ChineseTokenizer()
        >>> system = SparseRetrievalSystem(tokenizer)
        >>> system.load_documents("data_dir")
        >>> system.build_index()
        >>> results = system.search("人工智能", top_k=10)
    """

    def __init__(self,
                 tokenizer: Optional[ChineseTokenizer] = None,
                 k1: float = Config.BM25_K1,
                 b: float = Config.BM25_B):
        """
        初始化稀疏检索系统

        Args:
            tokenizer: 分词器实例，如果为None则创建默认分词器
            k1: BM25参数k1，控制词频饱和度
            b: BM25参数b，控制文档长度归一化程度
        """
        self.tokenizer = tokenizer or ChineseTokenizer(use_stopwords=Config.USE_STOPWORDS)

        # 文档集合
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.doc_tokens: Dict[int, List[str]] = {}

        # 索引结构
        # 倒排索引：term -> {doc_id: tf}
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_lengths: Dict[int, int] = {}
        self.avg_doc_length: float = 0.0

        # 词汇统计
        self.vocabulary: Set[str] = set()
        self.term_doc_freq: Dict[str, int] = defaultdict(int)

        # BM25参数
        self.k1 = k1
        self.b = b

        logger.info(f"稀疏检索系统初始化完成 (k1={k1}, b={b})")
        
    def _detect_encoding(self, file_path: Path) -> Optional[str]:
        """
        检测文件编码

        Args:
            file_path: 文件路径

        Returns:
            检测到的编码名称，如果检测失败返回None
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']

                # 只有置信度足够高才使用检测结果
                if encoding and confidence > 0.7:
                    logger.debug(f"检测到编码: {encoding} (置信度: {confidence:.2f})")
                    return encoding
                else:
                    logger.debug(f"编码检测置信度过低: {encoding} ({confidence:.2f})")
                    return None
        except Exception as e:
            logger.debug(f"编码检测失败: {e}")
            return None

    def read_file_with_encoding(self, file_path: Path) -> str:
        """
        使用正确的编码读取文件

        尝试多种编码方式读取文件，如果都失败则使用chardet自动检测。

        Args:
            file_path: 文件路径

        Returns:
            文件内容

        Raises:
            IOError: 如果文件无法读取
        """
        file_path = Path(file_path)

        # 首先尝试配置中的编码列表
        for encoding in Config.SUPPORTED_ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    if content.strip():
                        return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.error(f"读取文件失败 {file_path}: {e}")
                raise IOError(f"无法读取文件: {file_path}")

        # 如果所有预设编码都失败，使用chardet检测
        detected_encoding = self._detect_encoding(file_path)
        if detected_encoding:
            try:
                with open(file_path, 'r', encoding=detected_encoding) as f:
                    content = f.read()
                    if content.strip():
                        return content
            except Exception as e:
                logger.warning(f"使用检测编码 {detected_encoding} 读取失败: {e}")

        # 最后尝试忽略错误读取
        try:
            with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                content = f.read()
                if content.strip():
                    logger.warning(f"使用gbk+ignore模式读取文件: {file_path}")
                    return content
        except Exception as e:
            pass

        logger.error(f"无法读取文件 {file_path}")
        raise IOError(f"无法读取文件: {file_path}")
    
    def load_documents(self,
                      data_dir: Optional[Path] = None,
                      max_docs_per_category: Optional[int] = None) -> int:
        """
        从目录加载文档

        Args:
            data_dir: 数据目录路径，如果为None则使用配置中的路径
            max_docs_per_category: 每个类别加载的最大文档数，None表示加载全部

        Returns:
            成功加载的文档数量

        Raises:
            FileNotFoundError: 如果数据目录不存在
        """
        if data_dir is None:
            data_dir = Config.DATA_DIR
        else:
            data_dir = Path(data_dir)

        train_dir = data_dir / 'train'

        if not train_dir.exists():
            raise FileNotFoundError(f"训练数据目录不存在: {train_dir}")

        logger.info(f"开始加载文档 (目录: {train_dir})")

        # 获取所有类别
        categories = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        logger.info(f"发现 {len(categories)} 个类别: {', '.join(categories)}")

        doc_id = 0
        failed_files = []

        # 使用进度条
        category_progress = tqdm(categories, desc="加载类别", disable=not Config.SHOW_PROGRESS)

        for category in category_progress:
            cat_dir = train_dir / category
            files = sorted([f for f in cat_dir.iterdir() if f.suffix == '.txt'])

            # 限制每个类别的文档数
            if max_docs_per_category:
                files = files[:max_docs_per_category]

            category_progress.set_postfix({'类别': category, '文件数': len(files)})

            for file_path in files:
                try:
                    content = self.read_file_with_encoding(file_path)
                    if content and content.strip():
                        self.documents[doc_id] = {
                            'text': content,
                            'category': category,
                            'path': str(file_path),
                            'filename': file_path.name
                        }
                        doc_id += 1
                except Exception as e:
                    logger.warning(f"无法读取文件 {file_path}: {e}")
                    failed_files.append(str(file_path))

        logger.info(f"文档加载完成: 成功 {len(self.documents)} 个, 失败 {len(failed_files)} 个")

        if failed_files and len(failed_files) <= 10:
            logger.debug(f"失败的文件: {failed_files}")

        return len(self.documents)
    
    def build_index(self) -> None:
        """
        构建倒排索引

        对所有文档进行分词，构建倒排索引、文档长度索引等数据结构。

        索引结构：
            - inverted_index: {term: {doc_id: tf}}
            - doc_lengths: {doc_id: length}
            - vocabulary: {term}
            - term_doc_freq: {term: df}
        """
        if not self.documents:
            logger.warning("没有文档可以索引")
            return

        logger.info(f"开始构建索引 (文档数: {len(self.documents)})")
        start_time = time.time()

        total_length = 0

        # 使用进度条
        doc_progress = tqdm(
            self.documents.items(),
            desc="构建索引",
            total=len(self.documents),
            disable=not Config.SHOW_PROGRESS
        )

        for doc_id, doc_info in doc_progress:
            # 分词
            tokens = self.tokenizer.tokenize(doc_info['text'])
            self.doc_tokens[doc_id] = tokens

            # 计算词频
            term_freq = Counter(tokens)

            # 文档长度
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length

            # 构建倒排索引（优化：使用字典存储tf，避免后续遍历）
            for term, tf in term_freq.items():
                self.vocabulary.add(term)
                self.inverted_index[term][doc_id] = tf

                # 更新文档频率（只在第一次遇到该词时增加）
                if doc_id not in self.inverted_index[term] or len(self.inverted_index[term]) == 1:
                    self.term_doc_freq[term] += 1

            # 更新进度条信息
            if (doc_id + 1) % 100 == 0:
                doc_progress.set_postfix({
                    '词汇量': len(self.vocabulary),
                    '平均长度': f'{total_length / (doc_id + 1):.1f}'
                })

        # 计算平均文档长度
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0.0

        elapsed = time.time() - start_time

        logger.info(f"索引构建完成!")
        logger.info(f"  耗时: {elapsed:.2f}秒")
        logger.info(f"  文档数: {len(self.documents)}")
        logger.info(f"  词汇量: {len(self.vocabulary)}")
        logger.info(f"  平均文档长度: {self.avg_doc_length:.2f} 词")
        logger.info(f"  索引大小: {len(self.inverted_index)} 个词条")
        
    def calculate_idf(self, term: str) -> float:
        """
        计算词的IDF (Inverse Document Frequency)

        使用BM25的IDF公式：
            IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        Args:
            term: 词项

        Returns:
            IDF值
        """
        N = len(self.documents)
        df = self.term_doc_freq.get(term, 0)

        if df == 0:
            return 0.0

        # BM25的IDF公式
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf

    def calculate_bm25_score(self, query_terms: List[str], doc_id: int) -> float:
        """
        计算文档对查询的BM25相关性分数

        BM25公式：
            Score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1-b+b×|D|/avgdl))

        Args:
            query_terms: 查询词列表
            doc_id: 文档ID

        Returns:
            BM25分数
        """
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0)

        if doc_length == 0:
            return 0.0

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            # 获取词频（优化：直接从字典获取，不需要遍历）
            tf = self.inverted_index[term].get(doc_id, 0)

            if tf == 0:
                continue

            # 计算IDF
            idf = self.calculate_idf(term)

            # BM25公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)

        return score
    
    def search(self,
              query: str,
              top_k: int = Config.DEFAULT_TOP_K,
              min_score: float = Config.MIN_SCORE_THRESHOLD) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        检索相关文档

        Args:
            query: 查询字符串
            top_k: 返回的最大结果数
            min_score: 最小相关性分数阈值

        Returns:
            检索结果列表，格式为 [(doc_id, score, doc_info), ...]
            按相关性分数降序排列

        Examples:
            >>> results = system.search("人工智能技术", top_k=10)
            >>> for doc_id, score, doc_info in results:
            ...     print(f"{doc_info['filename']}: {score:.4f}")
        """
        if not query or not query.strip():
            logger.warning("查询为空")
            return []

        # 分词
        query_terms = self.tokenizer.tokenize(query)

        if not query_terms:
            logger.warning(f"查询分词后为空: {query}")
            return []

        logger.debug(f"查询: '{query}' -> 分词: {query_terms}")

        # 获取候选文档（包含至少一个查询词的文档）
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())

        if not candidate_docs:
            logger.info(f"未找到包含查询词的文档: {query_terms}")
            return []

        logger.debug(f"候选文档数: {len(candidate_docs)}")

        # 计算分数
        scores = []
        for doc_id in candidate_docs:
            score = self.calculate_bm25_score(query_terms, doc_id)

            # 过滤低分文档
            if score >= min_score:
                scores.append((doc_id, score, self.documents[doc_id]))

        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回Top-K结果
        top_k = min(top_k, Config.MAX_TOP_K)
        results = scores[:top_k]

        logger.info(f"查询 '{query}' 返回 {len(results)} 个结果")

        return results
    
    def save_index(self, filepath: Optional[Path] = None) -> None:
        """
        保存索引到文件

        Args:
            filepath: 索引文件路径，如果为None则使用配置中的路径
        """
        if filepath is None:
            filepath = Config.INDEX_FILE
        else:
            filepath = Path(filepath)

        logger.info(f"开始保存索引到: {filepath}")

        # 准备数据
        data = {
            'documents': self.documents,
            'doc_tokens': self.doc_tokens,
            'inverted_index': dict(self.inverted_index),
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'vocabulary': list(self.vocabulary),
            'term_doc_freq': dict(self.term_doc_freq),
            'k1': self.k1,
            'b': self.b,
            'metadata': {
                'num_documents': len(self.documents),
                'num_terms': len(self.vocabulary),
                'avg_doc_length': self.avg_doc_length,
                'timestamp': time.time()
            }
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"索引保存成功: {filepath} ({file_size:.2f} MB)")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise

    def load_index(self, filepath: Optional[Path] = None) -> None:
        """
        从文件加载索引

        Args:
            filepath: 索引文件路径，如果为None则使用配置中的路径

        Raises:
            FileNotFoundError: 如果索引文件不存在
        """
        if filepath is None:
            filepath = Config.INDEX_FILE
        else:
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"索引文件不存在: {filepath}")

        logger.info(f"开始加载索引: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # 恢复数据
            self.documents = data['documents']
            self.doc_tokens = data['doc_tokens']
            self.inverted_index = defaultdict(dict, data['inverted_index'])
            self.doc_lengths = data['doc_lengths']
            self.avg_doc_length = data['avg_doc_length']
            self.vocabulary = set(data['vocabulary'])
            self.term_doc_freq = defaultdict(int, data['term_doc_freq'])

            # 恢复参数
            if 'k1' in data:
                self.k1 = data['k1']
            if 'b' in data:
                self.b = data['b']

            logger.info(f"索引加载成功:")
            logger.info(f"  文档数: {len(self.documents)}")
            logger.info(f"  词汇量: {len(self.vocabulary)}")
            logger.info(f"  平均文档长度: {self.avg_doc_length:.2f}")
            logger.info(f"  BM25参数: k1={self.k1}, b={self.b}")

        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            raise


    def get_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            统计信息字典
        """
        stats = {
            'num_documents': len(self.documents),
            'num_terms': len(self.vocabulary),
            'avg_doc_length': self.avg_doc_length,
            'total_tokens': sum(self.doc_lengths.values()),
            'k1': self.k1,
            'b': self.b,
        }

        # 类别统计
        category_counts = defaultdict(int)
        for doc_info in self.documents.values():
            category_counts[doc_info['category']] += 1
        stats['categories'] = dict(category_counts)
        stats['num_categories'] = len(category_counts)

        return stats

    def print_statistics(self) -> None:
        """打印系统统计信息"""
        stats = self.get_statistics()

        print("\n" + "=" * 80)
        print("检索系统统计信息")
        print("=" * 80)
        print(f"文档总数: {stats['num_documents']}")
        print(f"词汇总量: {stats['num_terms']}")
        print(f"平均文档长度: {stats['avg_doc_length']:.2f} 词")
        print(f"总词数: {stats['total_tokens']}")
        print(f"类别数: {stats['num_categories']}")
        print(f"BM25参数: k1={stats['k1']}, b={stats['b']}")
        print("\n各类别文档数:")
        for category, count in sorted(stats['categories'].items()):
            print(f"  {category}: {count}")
        print("=" * 80)


def main():
    """主函数：构建检索系统"""
    logger.info("=" * 80)
    logger.info("稀疏检索系统 - 索引构建")
    logger.info("=" * 80)

    # 创建分词器
    tokenizer = ChineseTokenizer(
        use_stopwords=Config.USE_STOPWORDS,
        min_word_length=Config.MIN_WORD_LENGTH
    )

    # 创建检索系统
    retrieval_system = SparseRetrievalSystem(
        tokenizer=tokenizer,
        k1=Config.BM25_K1,
        b=Config.BM25_B
    )

    # 加载文档
    logger.info(f"加载文档 (每类别最多: {Config.MAX_DOCS_PER_CATEGORY or '全部'})")
    num_docs = retrieval_system.load_documents(
        data_dir=Config.DATA_DIR,
        max_docs_per_category=Config.MAX_DOCS_PER_CATEGORY
    )

    if num_docs == 0:
        logger.error("没有加载到任何文档，程序退出")
        return

    # 构建索引
    retrieval_system.build_index()

    # 打印统计信息
    retrieval_system.print_statistics()

    # 保存索引
    retrieval_system.save_index(Config.INDEX_FILE)

    logger.info("=" * 80)
    logger.info("索引构建完成!")
    logger.info("=" * 80)

    # 测试检索
    logger.info("\n测试检索功能:")
    test_query = "计算机网络技术"
    logger.info(f"查询: {test_query}")
    results = retrieval_system.search(test_query, top_k=5)

    for rank, (doc_id, score, doc_info) in enumerate(results, 1):
        logger.info(f"  {rank}. [{doc_info['category']}] {doc_info['filename']} (分数: {score:.4f})")


if __name__ == '__main__':
    main()

