#%%
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import json
import numpy as np
import re
from utils.common_use import llm_t0, embedding, extract_json
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# chunks 应该是用户提供的字典列表，包含chunk内容和timestamp
# chunks = [{'chunk': "文本内容", 'timestamp': '时间戳'}, ...]  # 新格式示例
# 兼容旧格式：chunks = ["文本1", "文本2", "文本3"]  # 旧格式示例

def extract_pure_chunks(chunks_with_time):
    """
    从带时间戳的chunks中提取纯文本内容，用于后续处理
    
    Args:
        chunks_with_time: list of chunks with timestamp (新格式) 或 pure text chunks (旧格式)
    
    Returns:
        list: 纯文本内容的chunks列表
    """
    pure_chunks = []
    for chunk in chunks_with_time:
        if isinstance(chunk, dict):
            pure_chunks.append(chunk.get('chunk', ''))
        elif isinstance(chunk, str):
            pure_chunks.append(chunk)
        else:
            pure_chunks.append(str(chunk))
    return pure_chunks

def extract_timestamps(chunks_with_time):
    """
    从带时间戳的chunks中提取时间戳列表
    
    Args:
        chunks_with_time: list of chunks with timestamp (新格式) 或 pure text chunks (旧格式)
    
    Returns:
        list: 时间戳列表，如果没有时间戳则为空字符串
    """
    timestamps = []
    for chunk in chunks_with_time:
        if isinstance(chunk, dict):
            timestamps.append(chunk.get('timestamp', ''))
        else:
            timestamps.append('')
    return timestamps

#1.节点的处理

#获得节点的summary

sum_prompt = """
You are a professional text summarization assistant. Read the text below and provide a concise summary. Output **only** valid JSON with a single field `summary`.

Text:
\"\"\"{input_text}\"\"\"

Output format:
{{
  "summary": "A clear and concise summary of the text"
}}

Rules:
1. Output must be valid JSON.
2. Do not include any extra text or comments.
"""


def generate_summaries(chunks):
    """为每个chunk生成summary"""
    print(f"正在为 {len(chunks)} 个chunks生成summary...")
    summaries = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"处理第 {i+1}/{len(chunks)} 个chunk的summary")
        prompt = sum_prompt.format(input_text=chunk)
        response = llm_t0("You are a helpful assistant.", prompt)
        result = extract_json(response)
        
        if result is None:
            print(f"第 {i} 个chunk的JSON解析失败，使用空字符串")
            summaries.append("")  # 如果解析失败，使用空字符串
        elif not isinstance(result, dict):
            print(f"第 {i} 个chunk的解析结果不是字典类型，使用空字符串")
            summaries.append("")
        elif 'summary' not in result:
            print(f"第 {i} 个chunk的解析结果缺少'summary'键，使用空字符串")
            summaries.append("")
        else:
            summaries.append(result['summary'])
    return summaries

#2.边的处理

#通过相似度矩阵构建边（embedding函数是embedding()）

def split_text_into_sentences(text):
    """
    自定义句子分割函数，支持中英文标点符号，并过滤长度小于30的句子
    支持对话格式，保留说话者信息
    
    Args:
        text: 输入文本（可以是字符串或对话行列表）
    
    Returns:
        list: 过滤后的句子列表（带说话者信息）
    """
    sentences = []
    
    # 如果输入是字符串，按行分割处理对话格式
    if isinstance(text, str):
        lines = text.split('\n')
    else:
        lines = text
    
    for line in lines:
        if not isinstance(line, str):
            continue
            
        line = line.strip()
        if not line:
            continue
        
        # 检查是否为对话格式 (speaker: content)
        match = re.match(r'^([^:]+):\s*(.*)$', line)
        if match:
            # 对话格式，保留说话者信息
            speaker, content = match.groups()
            
            # 按标点符号分割内容
            parts = re.split(r'([.?!。！？])', content)
            
            for i in range(0, len(parts) - 1, 2):
                sentence_text = parts[i].strip()
                punctuation = parts[i + 1]
                
                if not sentence_text:
                    continue
                
                # 重新组合句子，保留说话者信息
                sentence = f"{speaker}: {sentence_text}{punctuation}"
                
                # 过滤长度小于30的句子
                if len(sentence) >= 30:
                    sentences.append(sentence)
        else:
            # 非对话格式，直接按标点符号分割
            parts = re.split(r'([.?!。！？;；])', line)
            
            for i in range(0, len(parts) - 1, 2):
                sentence_text = parts[i].strip()
                punctuation = parts[i + 1]
                
                if not sentence_text:
                    continue
                
                sentence = f"{sentence_text}{punctuation}"
                
                # 过滤长度小于30的句子
                if len(sentence) >= 30:
                    sentences.append(sentence)
    
    return sentences


def compute_similarity_matrix(chunks):
    """计算chunks之间的相似度矩阵"""
    print("正在计算文本embedding和相似度矩阵...")
    # 获取所有chunks的embedding
    embeddings = embedding(chunks)
    embeddings_array = np.array(embeddings)
    
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(embeddings_array)
    print(f"相似度矩阵计算完成，形状: {similarity_matrix.shape}")
    return similarity_matrix

#根据句子相似度构建边

def build_sentence_similarity_edges(chunks, sentence_threshold=0.7):
    """基于句子相似度构建边"""
    print("正在构建句子相似度边...")
    
    # 对每个chunk进行句子分割
    all_sentences = []
    chunk_to_sentences = []  # 记录每个chunk包含的句子索引范围
    chunk_sentences_list = []  # 存储每个chunk的句子列表
    
    for i, chunk in enumerate(chunks):
        # 使用自定义句子分割函数，自动过滤长度小于30的句子
        sentences = split_text_into_sentences(chunk)
        chunk_sentences_list.append(sentences)
        
        start_idx = len(all_sentences)
        all_sentences.extend(sentences)
        end_idx = len(all_sentences)
        chunk_to_sentences.append((start_idx, end_idx))
    
    print(f"共提取了 {len(all_sentences)} 个句子（已过滤长度<30的句子）")
    
    if not all_sentences:
        return np.zeros((len(chunks), len(chunks))), chunk_sentences_list
    
    # 计算所有句子的embedding
    print("正在计算句子embedding...")
    sentence_embeddings = embedding(all_sentences)
    sentence_embeddings_array = np.array(sentence_embeddings)
    
    # 计算句子之间的相似度矩阵
    sentence_similarity_matrix = cosine_similarity(sentence_embeddings_array)
    print("句子相似度矩阵计算完成")
    
    # 构建chunk之间的句子相似度矩阵
    n_chunks = len(chunks)
    sentence_chunk_matrix = np.zeros((n_chunks, n_chunks))
    
    for i in range(n_chunks):
        for j in range(i + 1, n_chunks):
            start_i, end_i = chunk_to_sentences[i]
            start_j, end_j = chunk_to_sentences[j]
            
            # 计算chunk i和chunk j之间句子的最大相似度
            max_sim = 0
            for sent_i in range(start_i, end_i):
                for sent_j in range(start_j, end_j):
                    sim = sentence_similarity_matrix[sent_i][sent_j]
                    max_sim = max(max_sim, sim)
            
            sentence_chunk_matrix[i][j] = max_sim
            sentence_chunk_matrix[j][i] = max_sim
    
    # 统计超过阈值的边数
    edge_count = 0
    for i in range(n_chunks):
        for j in range(i + 1, n_chunks):
            if sentence_chunk_matrix[i][j] > sentence_threshold:
                edge_count += 1
    
    print(f"基于句子相似度构建了 {edge_count} 条潜在边")
    return sentence_chunk_matrix, chunk_sentences_list

#3.图构建



def build_complete_graph(chunks_with_time, similarity_threshold=0.7, sentence_threshold=0.8):
    """
    完整的图构建流程，构建两种独立的边类型
    
    Args:
        chunks_with_time: list of chunks with timestamp (新格式) 或 pure text chunks (旧格式)
        similarity_threshold: float, 整体文本相似度阈值
        sentence_threshold: float, 句子相似度阈值
    
    Returns:
        networkx.Graph: 构建好的图
    """
    print("=" * 50)
    print("开始构建完整图...")
    print("=" * 50)
    
    # 提取纯文本内容和时间戳
    pure_chunks = extract_pure_chunks(chunks_with_time)
    timestamps = extract_timestamps(chunks_with_time)
    
    print("步骤1: 生成节点的summary...")
    summaries = generate_summaries(pure_chunks)
    
    print("\n步骤2: 计算整体文本相似度矩阵...")
    similarity_matrix = compute_similarity_matrix(pure_chunks)
    
    print("\n步骤3: 构建句子相似度边...")
    sentence_matrix, chunk_sentences_list = build_sentence_similarity_edges(pure_chunks, sentence_threshold)
    
    print("\n步骤4: 构建图...")
    # 创建图
    G = nx.Graph()
    
    # 添加节点（包含所有属性）
    for i, (content, summary, sentences, timestamp) in enumerate(zip(pure_chunks, summaries, chunk_sentences_list, timestamps)):
        G.add_node(i, 
                  content=content, 
                  summary=summary,
                  sentences=sentences,
                  timestamp=timestamp,
                  node_type='segment')
    
    # 添加边 - 两种独立的边类型
    n = len(pure_chunks)
    
    text_similarity_edges = 0
    sentence_similarity_edges = 0
    both_type_edges = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # 检查整体文本相似度
            text_sim = similarity_matrix[i][j]
            sentence_sim = sentence_matrix[i][j]
            
            # 判断是否需要添加边
            add_edge = False
            edge_type = ""
            weight = 0
            
            if text_sim > similarity_threshold and sentence_sim > sentence_threshold:
                # 两种条件都满足
                add_edge = True
                edge_type = "both"
                weight = max(text_sim, sentence_sim)  # 使用较高的相似度作为权重
                both_type_edges += 1
            elif text_sim > similarity_threshold:
                # 只有整体文本相似度满足
                add_edge = True
                edge_type = "text_similarity"
                weight = text_sim
                text_similarity_edges += 1
            elif sentence_sim > sentence_threshold:
                # 只有句子相似度满足
                add_edge = True
                edge_type = "sentence_similarity"
                weight = sentence_sim
                sentence_similarity_edges += 1
            
            if add_edge:
                G.add_edge(i, j, 
                          weight=weight,
                          edge_type=edge_type,
                          text_similarity=text_sim,
                          sentence_similarity=sentence_sim)
    
    print(f"\n图构建完成！")
    print(f"节点数: {G.number_of_nodes()}")
    print(f"总边数: {G.number_of_edges()}")
    print(f"- 基于整体文本相似度的边: {text_similarity_edges}")
    print(f"- 基于句子相似度的边: {sentence_similarity_edges}")
    print(f"- 两种类型都满足的边: {both_type_edges}")
    print("=" * 50)
    
    return G

def build_simple_graph(chunks_with_time, threshold=0.8):
    """
    简化版图构建，只使用文本相似度
    
    Args:
        chunks_with_time: list of chunks with timestamp (新格式) 或 pure text chunks (旧格式)
        threshold: similarity threshold
    
    Returns:
        networkx.Graph: 构建好的图
    """
    print("构建简化版图（仅文本相似度）...")
    
    # 提取纯文本内容和时间戳
    pure_chunks = extract_pure_chunks(chunks_with_time)
    timestamps = extract_timestamps(chunks_with_time)
    
    # 计算相似度矩阵
    similarity_matrix = compute_similarity_matrix(pure_chunks)
    
    # 构建图
    G = nx.Graph()
    
    # 添加节点（包含时间戳）
    for i, (content, timestamp) in enumerate(zip(pure_chunks, timestamps)):
        G.add_node(i, 
                  content=content, 
                  timestamp=timestamp,
                  node_type='segment')
    
    # 添加边
    n = len(pure_chunks)
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(i, j, 
                          weight=similarity_matrix[i][j],
                          edge_type='similarity')
                edge_count += 1
    
    print(f"简化版图构建完成！节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    return G

# 使用示例
# if __name__ == "__main__":
#     # 示例数据
#     sample_chunks = [
#         "人工智能是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的机器。",
#         "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
#         "深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。",
#         "自然语言处理是人工智能的一个应用领域，专注于计算机与人类语言之间的交互。",
#         "计算机视觉是人工智能的另一个重要应用，使机器能够理解和解释视觉信息。"
#     ]
    
#     # 构建完整图（包含两种边类型）
#     graph = build_complete_graph(sample_chunks, similarity_threshold=0.5, sentence_threshold=0.6)
    
    # 或者构建简化版图
    # simple_graph = build_simple_graph(sample_chunks, threshold=0.6)

#%%