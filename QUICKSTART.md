# EventGraphRAG 快速开始指南

本指南将帮助您在5分钟内快速上手EventGraphRAG项目。

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/EventGraphRAG.git
cd EventGraphRAG

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API

```bash
# 复制配置文件
cp .env.example .env

# 编辑配置文件，填入你的OpenAI API密钥
# OPENAI_API_KEY=your_api_key_here
# OPENAI_BASE_URL=your_base_url_here
```

### 3. 运行第一个实验

```python
# 直接运行主程序
python main.py
```

## 📊 核心功能演示

### 事件切分

```python
from core.event_segments import event_segment_main

# 切分对话为事件
conversation_id = 6
chunks = event_segment_main(conversation_id)
print(f"切分得到 {len(chunks)} 个事件块")
```

### 图构建

```python
from core.graphconstraction import build_complete_graph

# 构建事件图
graph = build_complete_graph(chunks)
print(f"图包含 {graph.number_of_nodes()} 个节点，{graph.number_of_edges()} 条边")
```

### TOG检索

```python
from core.retrival_TOG import TOG_main

# 执行TOG检索
retrival_results = TOG_main(chunks, graph, conversation_id)
print(f"检索到 {len(retrival_results)} 个相关事件")
```

### 答案生成与评分

```python
from core.answer_rate import get_answer_and_rate

# 生成答案并评分
llmanswerlist, rate_all = get_answer_and_rate(retrival_results, conversation_id)
print(f"平均评分: {rate_all['average']:.1f}")
```

## 🎯 实验对比

### 与基线方法对比

```python
from core.event_segments import get_turn_chunks_withtime
from core.retrival_others import topk_retival

# Turn-based方法
turn_chunks = get_turn_chunks_withtime(conversation_id)
turn_results = topk_retival(turn_chunks, topK=6, conversation_id=conversation_id)
turn_score = get_answer_and_rate(turn_results, conversation_id)[1]['average']

print(f"EventGraphRAG: {rate_all['average']:.1f}")
print(f"Turn-based: {turn_score:.1f}")
```

### 消融实验

```python
import networkx as nx

# 移除特定类型的边
def graph_without_edge_type(g: nx.Graph, edge_type: str):
    g_new = g.copy()
    edges_to_remove = [
        (u, v) for u, v, attrs in g_new.edges(data=True)
        if attrs.get('edge_type') == edge_type
    ]
    g_new.remove_edges_from(edges_to_remove)
    return g_new

# 测试不同图结构
graph_wotext = graph_without_edge_type(graph, 'text_similarity')
results_wotext = TOG_main(chunks, graph_wotext, conversation_id)
score_wotext = get_answer_and_rate(results_wotext, conversation_id)[1]['average']

print(f"完整图: {rate_all['average']:.1f}")
print(f"无文本相似边: {score_wotext:.1f}")
```

## 📈 结果分析

### 查看详细结果

```python
# 查看不同问题类型的性能
print("各问题类型评分:")
for question_type, score in rate_all.items():
    if question_type != 'average':
        print(f"  {question_type}: {score:.1f}")

# 查看具体答案
print("\n示例答案:")
for i, answer in enumerate(llmanswerlist[:2]):
    print(f"Q{i+1}: {answer.get('question', 'N/A')}")
    print(f"A: {answer.get('answer', 'N/A')[:100]}...")
```

### 可视化结果

```python
import matplotlib.pyplot as plt

# 简单的性能对比图
methods = ['EventGraphRAG', 'Turn-based', 'Session-based']
scores = [rate_all['average'], turn_score, session_score]

plt.figure(figsize=(8, 5))
plt.bar(methods, scores, color=['blue', 'green', 'orange'])
plt.ylabel('平均评分')
plt.title('不同方法性能对比')
plt.show()
```

## 🔧 自定义配置

### 修改切分策略

```python
from core.event_segments import get_fixsize_chunks_withtime

# 使用固定大小切分
fixed_chunks = get_fixsize_chunks_withtime(conversation_id, chunk_size=200)
print(f"固定大小切分: {len(fixed_chunks)} 个块")
```

### 修改检索参数

```python
from core.retrival_others import topk_retival

# 调整Top-K值
top5_results = topk_retival(chunks_pure, topK=5, conversation_id=conversation_id)
top10_results = topk_retival(chunks_pure, topK=10, conversation_id=conversation_id)
```

## 📝 保存和加载结果

```python
import json
from utils.common_use import save_class, load_class

# 保存结果
with open('my_results.json', 'w', encoding='utf-8') as f:
    json.dump(rate_all, f, ensure_ascii=False, indent=2)

# 保存图结构
save_class(graph, 'my_graph.pkl')

# 加载图结构
loaded_graph = load_class('my_graph.pkl')
```

## 🐛 常见问题

### Q: API调用失败怎么办？
A: 检查`.env`文件中的API密钥是否正确，确认网络连接正常。

### Q: 内存不足怎么办？
A: 减少处理的数据量，或者使用更小的模型。

### Q: 结果不一致怎么办？
A: 设置随机种子，确保实验的可重现性。

```python
import random
import numpy as np

# 设置随机种子
random.seed(42)
np.random.seed(42)
```

## 📚 更多资源

- **完整文档**: [README.md](README.md)
- **实验复现**: [EXPERIMENTS.md](EXPERIMENTS.md)
- **贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **更新日志**: [CHANGELOG.md](CHANGELOG.md)

## 🎉 下一步

1. **深入了解**: 阅读[EXPERIMENTS.md](EXPERIMENTS.md)了解完整实验流程
2. **自定义实验**: 修改参数进行自己的实验
3. **贡献代码**: 查看[CONTRIBUTING.md](CONTRIBUTING.md)了解如何贡献
4. **引用论文**: 如果在研究中使用EventGraphRAG，请引用我们的论文

---

**💡 提示**: 如果遇到问题，请查看[EXPERIMENTS.md](EXPERIMENTS.md)中的故障排除部分，或在GitHub上提交Issue。