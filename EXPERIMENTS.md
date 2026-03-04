# 实验复现指南

本指南详细说明了如何复现EventGraphRAG论文中的实验结果。

## 实验环境

### 系统要求
- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 至少16GB RAM
- 至少10GB可用磁盘空间

### 依赖安装

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

### 环境配置

1. 复制环境变量文件：
```bash
cp .env.example .env
```

2. 编辑`.env`文件，填入你的OpenAI API配置：
```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_base_url_here
```

## 数据准备

### 数据集说明

实验使用LocoMo10数据集，包含多轮对话和相应的问答对。

### 下载数据

```bash
# 数据下载脚本（示例）
python scripts/download_data.py
```

### 数据预处理

```python
from data.preprocessing import preprocess_sessions_with_rounds
import json

# 加载原始数据
with open('data/raw/locomo10.json', 'r') as f:
    raw_data = json.load(f)

# 预处理数据
processed_data = preprocess_sessions_with_rounds(raw_data, rounds=1)

# 保存处理后的数据
with open('data/processed/locomo10_processed.json', 'w') as f:
    json.dump(processed_data, f)
```

## 基础实验复现

### 1. 主实验 (EventGraphRAG)

```python
# 运行完整实验流程
python main.py
```

### 2. 基于块的方法对比

#### Turn-based方法
```python
from core.event_segments import get_turn_chunks_withtime
from core.retrival_others import topk_retival
from core.answer_rate import get_answer_and_rate

conversation_id = 6
chunks = get_turn_chunks_withtime(conversation_id)
retrival_results = topk_retival(chunks, topK=6, conversation_id=conversation_id)
llmanswerlist, rate_all = get_answer_and_rate(retrival_results, conversation_id)
```

#### Session-based方法
```python
from core.event_segments import get_session_chunks_withtime

chunks = get_session_chunks_withtime(conversation_id)
# 其余步骤同上
```

#### Fixed-size方法
```python
from core.event_segments import get_fixsize_chunks_withtime

# 不同大小的固定块
for size in [100, 200, 300]:
    chunks = get_fixsize_chunks_withtime(conversation_id, size)
    # 其余步骤同上
```

### 3. HippoRAG2基线

```python
from core.retrival_others import hipporag_retrival

retrival_results = hipporag_retrival(chunks, conversation_id)
llmanswerlist, rate_all = get_answer_and_rate(retrival_results, conversation_id)
```

## 消融实验

### 1. 事件切分消融

```python
from core.event_segments import (
    event_segment_main,           # 完整事件切分
    get_turn_chunks_withtime,      # Turn切分
    get_session_chunks_withtime,   # Session切分
    get_fixsize_chunks_withtime,   # 固定大小切分
    event_segment_simple          # 简单提示切分
)

conversation_id = 2
methods = {
    'event_segment': event_segment_main(conversation_id),
    'turn': get_turn_chunks_withtime(conversation_id),
    'session': get_session_chunks_withtime(conversation_id),
    'fixsize_200': get_fixsize_chunks_withtime(conversation_id, 200),
    'simple_prompt': event_segment_simple(conversation_id)
}

results = {}
for method_name, chunks in methods.items():
    graph = build_complete_graph(chunks)
    retrival_results = TOG_main(chunks, graph, conversation_id)
    llmanswerlist, rate_all = get_answer_and_rate(retrival_results, conversation_id)
    results[method_name] = rate_all
```

### 2. 图结构消融

```python
import networkx as nx

def graph_without_edge_type(g: nx.Graph, edge_type: str):
    """移除指定类型的边"""
    g_new = g.copy()
    edges_to_remove = [
        (u, v) for u, v, attrs in g_new.edges(data=True)
        if attrs.get('edge_type') == edge_type
    ]
    g_new.remove_edges_from(edges_to_remove)
    return g_new

# 消融整体语义边
graph_wotext = graph_without_edge_type(graph, 'text_similarity')
# 消融元素边
graph_wosentence = graph_without_edge_type(graph, 'sentence_similarity')

# 测试不同图结构
for graph_name, test_graph in [
    ('full_graph', graph),
    ('without_text_similarity', graph_wotext),
    ('without_sentence_similarity', graph_wosentence)
]:
    retrival_results = TOG_main(chunks, test_graph, conversation_id)
    llmanswerlist, rate_all = get_answer_and_rate(retrival_results, conversation_id)
```

### 3. 检索方法消融

```python
from core.retrival_others import (
    topk_retival,      # Top-K检索
    ppr_retival        # Personalized PageRank检索
)

# Top-K检索
chunks_pure = [f"time:{i['timestamp']} content:{i['chunk']}" for i in chunks]
retrival_results_topk = topk_retival(chunks_pure, topK=5, conversation_id=conversation_id)

# PPR检索
retrival_results_ppr = ppr_retival(chunks_pure, graph, alpha=0.85, max_iter=6)

# 无TOG检索
retrival_results_wotog = [i[0:3] for i in retrival_results]
```

## 性能评估

### 1. 运行单个实验

```python
# 完整实验流程
def run_full_experiment(conversation_id):
    # 事件切分
    chunks = event_segment_main(conversation_id)
    
    # 图构建
    graph = build_complete_graph(chunks)
    
    # TOG检索
    retrival_results = TOG_main(chunks, graph, conversation_id)
    
    # 答案生成与评分
    llmanswerlist, rate_all = get_answer_and_rate(retrival_results, conversation_id)
    
    return {
        'chunks': chunks,
        'graph': graph,
        'retrival_results': retrival_results,
        'llmanswerlist': llmanswerlist,
        'rate_all': rate_all
    }

# 运行实验
results = run_full_experiment(conversation_id=6)
```

### 2. 批量实验

```python
import os
import json

def batch_experiment(conversation_ids):
    all_results = {}
    
    for conv_id in conversation_ids:
        print(f"处理对话 {conv_id}...")
        try:
            results = run_full_experiment(conv_id)
            all_results[conv_id] = results
            
            # 保存中间结果
            os.makedirs(f'output/main_results/conversation_index_{conv_id}', exist_ok=True)
            
            # 保存各个组件的结果
            with open(f'output/main_results/conversation_index_{conv_id}/chunks.json', 'w') as f:
                json.dump(results['chunks'], f, ensure_ascii=False, indent=2)
            
            # 保存图结构
            save_class(results['graph'], f'output/main_results/conversation_index_{conv_id}/graph.pkl')
            
            # 保存检索结果和评分
            with open(f'output/main_results/conversation_index_{conv_id}/retrival_results.json', 'w') as f:
                json.dump(results['retrival_results'], f, ensure_ascii=False, indent=2)
            
            with open(f'output/main_results/conversation_index_{conv_id}/llmanswerlist.json', 'w') as f:
                json.dump(results['llmanswerlist'], f, ensure_ascii=False, indent=2)
            
            with open(f'output/main_results/conversation_index_{conv_id}/rate_all.json', 'w') as f:
                json.dump(results['rate_all'], f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"处理对话 {conv_id} 时出错: {e}")
            continue
    
    return all_results

# 运行批量实验
conversation_ids = [3, 6]  # 示例对话ID
batch_results = batch_experiment(conversation_ids)
```

## 结果分析

### 1. 性能统计

```python
import pandas as pd
import numpy as np

def analyze_performance(results_dict):
    """分析实验结果"""
    data = []
    
    for conv_id, results in results_dict.items():
        rate_all = results['rate_all']
        
        # 提取不同问题类型的分数
        data.append({
            'conversation_id': conv_id,
            'single_hop': rate_all.get('single_hop', 0),
            'multi_hop': rate_all.get('multi_hop', 0),
            'temporal': rate_all.get('temporal', 0),
            'open_domain': rate_all.get('open_domain', 0),
            'average': rate_all.get('average', 0)
        })
    
    df = pd.DataFrame(data)
    return df

# 分析结果
df = analyze_performance(batch_results)
print("性能统计:")
print(df.describe())
```

### 2. 结果可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(df):
    """可视化实验结果"""
    plt.figure(figsize=(12, 8))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制性能对比图
    metrics = ['single_hop', 'multi_hop', 'temporal', 'open_domain']
    
    for metric in metrics:
        plt.plot(df['conversation_id'], df[metric], marker='o', label=metric)
    
    plt.xlabel('对话ID')
    plt.ylabel('分数')
    plt.title('EventGraphRAG性能表现')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制结果图
plot_results(df)
```

## 预期运行时间

| 实验类型 | 预计时间 | 说明 |
|----------|----------|------|
| 单个对话完整实验 | 5-15分钟 | 取决于对话长度和API响应速度 |
| 消融实验（所有对话） | 1-2小时 | 包含多种变体的实验 |
| 批量实验（10个对话） | 1-3小时 | 并行处理可以缩短时间 |

## 故障排除

### 常见问题

1. **API调用失败**
   - 检查API密钥是否正确
   - 确认网络连接
   - 检查API配额

2. **内存不足**
   - 减少批量处理大小
   - 使用更小的模型
   - 增加交换空间

3. **结果不一致**
   - 检查随机种子设置
   - 确认模型版本一致
   - 验证数据预处理步骤

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 保存中间结果
save_intermediate_results = True

# 减少数据规模进行测试
debug_mode = True
if debug_mode:
    conversation_ids = [3]  # 只测试一个对话
```

## 引用

如果您使用本代码复现实验，请引用：

```bibtex
@misc{eventgraphrag2024,
  title={EventGraphRAG: Event-based Long-term Memory Framework for Conversational AI},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/EventGraphRAG}
}