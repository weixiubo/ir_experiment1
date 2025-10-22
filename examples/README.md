# 示例代码

本目录包含了BM25中文检索系统的使用示例。

## 示例列表

### 1. basic_usage.py - 基本使用示例

演示如何使用检索系统进行基本的文档检索。

**运行方式：**
```bash
cd examples
python basic_usage.py
```

**功能：**
- 创建检索系统实例
- 构建索引
- 执行多个查询
- 显示检索结果

---

### 2. parameter_tuning.py - 参数调优示例

演示如何通过网格搜索找到最佳的BM25参数。

**运行方式：**
```bash
cd examples
python parameter_tuning.py
```

**功能：**
- 网格搜索k1和b参数
- 评估不同参数组合的性能
- 找到最佳参数组合
- 使用最佳参数进行检索

---

## 注意事项

1. 运行示例前请确保已安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 确保数据集已下载并解压到 `data/复旦中文文本分类语料库/train/` 目录

3. 首次运行会构建索引，可能需要几分钟时间

4. 索引文件会保存在 `output/index/` 目录，后续运行会自动加载

## 自定义示例

你可以基于这些示例创建自己的检索应用。主要步骤：

1. 导入检索系统：
   ```python
   from sparse_retrieval_system import SparseRetrievalSystem
   ```

2. 创建实例并构建索引：
   ```python
   system = SparseRetrievalSystem(data_dir='path/to/data')
   system.build_index()
   ```

3. 执行检索：
   ```python
   results = system.search('查询词', top_k=10)
   ```

更多详细信息请参考主目录的 README.md 和 使用指南.md。

