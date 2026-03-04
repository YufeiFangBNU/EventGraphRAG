#%%
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import json
import data.preprocessing as prep
import utils.common_use as common_use

import numpy as np
file_path='data/locomo10.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Token计算相关类和函数
import tiktoken

class TokenTracker:
    """Token使用统计器"""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.call_count = 0
    
    def add_call(self, input_tokens, output_tokens):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.call_count += 1
    
    def get_report(self):
        return {
            'total_input_tokens': self.input_tokens,
            'total_output_tokens': self.output_tokens,
            'total_tokens': self.input_tokens + self.output_tokens,
            'call_count': self.call_count,
            'avg_tokens_per_call': (self.input_tokens + self.output_tokens) / max(1, self.call_count)
        }
    
    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.call_count = 0

def count_tokens_gpt4(text, model="gpt-4"):
    """使用tiktoken精确计算tokens"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # 如果模型不支持，使用gpt-4的编码
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def llm_with_token_count(sys, context, model='gpt-4o', tracker=None):
    """
    包装llm_t0函数，添加token计数功能
    返回: (response, input_tokens, output_tokens)
    """
    # 计算输入tokens
    sys_text = str(sys)
    context_text = str(context)
    input_tokens = count_tokens_gpt4(sys_text) + count_tokens_gpt4(context_text)
    
    # 调用原始LLM函数
    response = common_use.llm_t0(sys, context, model)
    
    # 计算输出tokens
    output_tokens = count_tokens_gpt4(response)
    
    # 记录到tracker
    if tracker:
        tracker.add_call(input_tokens, output_tokens)
    
    return response, input_tokens, output_tokens

#获取turn单元信息
def get_turn_chunks(data):
    turn_chunks=prep.preprocess_sessions_with_rounds_no_timestamp(raw=data, rounds=1)#可在这里设置切分的round数，一个round代表一来一回
    turn_chunks_all=[]
    for i in turn_chunks:
        turn_chunks_all.extend(i)
    return turn_chunks_all
#获取session单元信息

def get_turn_chunks_withtime(conversation_id):
    turn_chunks=get_turn_chunks(data[conversation_id]['conversation'])
    turn_chunks_withtime=prep.get_chunks_with_timestamps(turn_chunks,data[conversation_id]['conversation'])
    return turn_chunks_withtime

def get_session_chunks(data):
    session_chunks=prep.preprocess_sessions_with_rounds_no_timestamp(raw=data, rounds=100)#获取整段对话为列表元素的信息
    session_chunks_all=[]
    for i in session_chunks:
        session_chunks_all.extend(i)
    return session_chunks_all

def get_session_chunks_withtime(conversation_id):
    session_chunks=get_session_chunks(data[conversation_id]['conversation'])
    session_chunks_withtime=prep.get_chunks_with_timestamps(session_chunks,data[conversation_id]['conversation'])
    return session_chunks_withtime


#获取单句为列表元素的信息
def get_sentence_chunks_flat(data):
    sentence_chunks=prep.preprocess_dialogue_to_single(data)
    sentence_chunks_all=[]
    for i in sentence_chunks:
        sentence_chunks_all.extend(i)
    return sentence_chunks_all

def get_sentence_chunks(data):
    sentence_chunks=prep.preprocess_dialogue_to_single(data)
    sentence_chunks_all=[]
    for i in sentence_chunks:
        sentence_chunks_all.append(i)
    return sentence_chunks_all



#fixsize切分函数，输入chunks_all
def merge_strings_with_limit(strings, k):
    result = []
    current = ""

    for s in strings:
        # 如果 current 为空，直接放
        if not current:
            current = s
        # 否则尝试加上 '\n' + s
        elif len(current) + 1 + len(s) <= k:
            current += "\n" + s
        else:
            # 超过限制，先保存当前结果
            result.append(current)
            current = s

    # 循环结束后别忘了把最后一个加进去
    if current:
        result.append(current)

    return result
#得到固定token长度的chunks，输入chunks_all
def get_fixsize_chunks(strings, k):
    result = []
    current = ""
    current_tokens = 0

    for s in strings:
        s_tokens = count_tokens_gpt4(s,'gpt-4')  # 计算当前字符串的tokens

        if not current:
            current = s
            current_tokens = s_tokens
        elif current_tokens + s_tokens <= k:
            current += "\n" + s
            current_tokens += s_tokens
        else:
            result.append(current)
            current = s
            current_tokens = s_tokens

    if current:
        result.append(current)

    return result

def get_fixsize_chunks_withtime(conversation_id,k):
    sentence_chunks=get_sentence_chunks_flat(data[conversation_id]['conversation'])
    fixsize_chunks = get_fixsize_chunks(sentence_chunks,k)  
    fixsize_chunks_withtime = prep.get_chunks_with_timestamps(fixsize_chunks, data[conversation_id]['conversation'])
    return fixsize_chunks_withtime
#eventpredict


import re
from typing import List

def split_into_sentences_filterd(dialog):
    """
    Returns:
        sentences: List[str]
            切分后的 sentence 文本（带 speaker）
        mapping: List[int]
            与 sentences 等长，表示每个 sentence 来源的原文 index
    """

    sentences = []
    mapping = []

    for idx, line in enumerate(dialog):
        if not isinstance(line, str):
            continue

        match = re.match(r'^([^:]+):\s*(.*)$', line)
        if not match:
            if len(line) >= 30:          # 👈 新规则
                sentences.append(line)
                mapping.append(idx)
            continue

        speaker, content = match.groups()
        parts = re.split(r'([.?!？])', content)

        for i in range(0, len(parts) - 1, 2):
            text = parts[i].strip()
            punct = parts[i + 1]

            if not text:
                continue

            sentence = f"{speaker}:{text}{punct}"

            if len(sentence) < 30:       # 👈 新规则
                continue

            sentences.append(sentence)
            mapping.append(idx)

    return sentences, mapping

def split_into_sentences(dialog):
    """
    Returns:
        sentences: List[str]
            切分后的 sentence 文本（带 speaker）
        mapping: List[int]
            与 sentences 等长，表示每个 sentence 来源的原文 index
    """

    sentences = []
    mapping = []

    for idx, line in enumerate(dialog):
        if not isinstance(line, str):
            continue

        match = re.match(r'^([^:]+):\s*(.*)$', line)
        if not match:
            sentences.append(line)
            mapping.append(idx)
            continue

        speaker, content = match.groups()
        parts = re.split(r'([.?!？])', content)

        for i in range(0, len(parts) - 1, 2):
            text = parts[i].strip()
            punct = parts[i + 1]

            if not text:
                continue

            sentence = f"{speaker}:{text}{punct}"
            sentences.append(sentence)
            mapping.append(idx)   # 👈 建立映射

    return sentences, mapping

predict_prompt='''Predict the next two dialogue turns based on the given dialogue history.

Input:
[
  "1:56 pm on 8 May, 2023",
  "SpeakerA: ...",
  "SpeakerB: ...",
  "SpeakerA: ...",
  "SpeakerB: ..."
]

Output:
speakA: 
speakB: 

'''

# predict_prompt_single = '''Predict the next dialogue turn based on the given dialogue history.

# The predicted turn should be said by the next speaker in alternation:
# - If the last turn was by SpeakerA, predict SpeakerB.
# - If the last turn was by SpeakerB, predict SpeakerA.

# Input:
# [
#   "SpeakerA: ...",
#   "SpeakerB: ...",
#   "SpeakerA: ...",
#   "SpeakerB: ..."
# ]

# Output:

# '''


predict_prompt_single = '''Predict the next dialogue turn based on the given dialogue history.


Input:
[
  "SpeakerA: ...",
  "SpeakerB: ...",
  "SpeakerA: ...",
  "SpeakerB: ..."
]

Output:

'''
def event_predict(context, tracker=None):
    pre, input_tokens, output_tokens = llm_with_token_count(predict_prompt_single, context, model='gpt-4o', tracker=tracker)
    return pre




def predict_segment(chunks, tracker=None):
    demosession=chunks
    demosession_text=[]
    for i in range(demosession.__len__()):
        demosession_text.append(demosession[0:i+1])

    pre_list=[] 
    for i in range(len(demosession_text)):

        #print("Output:")
        pre=event_predict(demosession_text[i], tracker)
        #print(pre)
        pre_list.append(pre)

    pre_embeddings = np.array(common_use.embedding(pre_list))
    demo_embeddings = np.array(common_use.embedding(demosession))

    # =========================
    # 2️⃣ 计算 similarity_list
    # =========================
    similarity_list = []

    for i in range(len(pre_list)-1):
        # 计算 i+1 到 i+3 的索引（不要越界）
        end_idx = min(i+4, len(demosession))  # i+4 因为 range 是开区间
        sims = []

        for j in range(i+1, end_idx):
            # 使用预计算好的 embedding
            emb1 = pre_embeddings[i]
            emb2 = demo_embeddings[j]

            sim = float(np.dot(emb1, emb2.T))  
            sims.append(sim)

        mean_sim = sum(sims) / len(sims)
        similarity_list.append(mean_sim)

    
    return similarity_list

def post_process(similarity_list,chunks,mapping):
    
    smoothed=np.convolve(similarity_list, np.ones(3)/3, mode='same')
    from scipy.signal import find_peaks
    valleys, props = find_peaks(
        -smoothed,
        # prominence=0.1,
        # distance=5,
        # width=2
        prominence=0.05,
        distance=3,
        width=2

    )
    
    sentence_turn_idxs = valleys + 1
    dialog_turn_idxs = [mapping[i] for i in sentence_turn_idxs]
    boundaries=dialog_turn_idxs
    
    split_chunks = []
    start_idx = 0

    for b in boundaries:
        # 前一个 chunk 包含 boundary_index
        # chunk_sentences = chunks[start_idx:b+1]
         # 前一个 chunk 不包含 boundary_index
        chunk_sentences = chunks[start_idx:b]
        split_chunks.append("\n".join(chunk_sentences))
        # 下一段从 boundary_index 开始
        start_idx = b

    # 添加最后一段
    if start_idx < len(chunks):
        split_chunks.append("\n".join(chunks[start_idx:]))
    return split_chunks

#串行
# similarity_list_all=[]
# predicted_chunks_all=[]
# for i in chunks:
#     sentences, mapping = split_into_sentences(i)
#     similarity_list=predict_segment(sentences)
#     predicted_chunks=post_process(similarity_list,i,mapping)
#     similarity_list_all.append(similarity_list)     
#     predicted_chunks_all.extend(predicted_chunks)


#并行
def get_prected_chunks(chunks):
    from concurrent.futures import ThreadPoolExecutor

    # 创建token tracker
    token_tracker = TokenTracker()

    def process_chunk(i):
        sentences, mapping = split_into_sentences_filterd(i)
        similarity_list = predict_segment(sentences, token_tracker)
        predicted_chunks = post_process(similarity_list, i, mapping)
        print("done")
        return similarity_list, predicted_chunks


    similarity_list_all = []
    predicted_chunks_all = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(process_chunk, chunks)

    for similarity_list, predicted_chunks in results:
        similarity_list_all.append(similarity_list)
        predicted_chunks_all.extend(predicted_chunks)
    
    # 输出token统计
    token_report = token_tracker.get_report()
    print(f"\n=== get_prected_chunks Token消耗统计 ===")
    print(f"总输入tokens: {token_report['total_input_tokens']}")
    print(f"总输出tokens: {token_report['total_output_tokens']}")
    print(f"总tokens: {token_report['total_tokens']}")
    print(f"调用次数: {token_report['call_count']}")
    print(f"平均每次调用tokens: {token_report['avg_tokens_per_call']:.2f}")
    print("=" * 50)
    
    return similarity_list_all, predicted_chunks_all, token_report



change_prompt_simple='''You are a cognitive science and language modeling research assistant.

Below is a continuous dialogue presented as indexed utterances (including image insertion markers). The dialogue has not been pre-segmented and should be treated as a single interaction stream.

Your task is to:

Identify indices where the main topic of conversation changes, and provide a brief explanation for each change.

A "topic change" is defined as a point where the core subject or focus of the dialogue shifts, such that the subsequent utterances no longer naturally continue the previous topic. This can include:

- A change in the person, activity, or event being discussed
- A shift in the central theme or focus of the conversation
- A natural transition from one discussion cluster to another

Do NOT mark boundaries solely based on:
- Speaker turn-taking
- Sentence endings
- The presence of an image marker alone

Output Format:

Return a JSON object containing a list `topic_boundaries`, where each element is an object with:

- `boundary_index`: the index of the topic change
- `reason`: a brief explanation describing why this point marks a topic change

Example:

```json
{
  "topic_boundaries": [
    {
      "boundary_index": 2,
      "reason": "The conversation shifts from discussing weekend plans to talking about favorite books."
    },
    {
      "boundary_index": 5,
      "reason": "The conversation shifts from books to planning a group hiking trip."
    }
  ]
}
'''

situation_model_prompt='''
You are given a dialogue consisting of indexed utterances.
Your task is to identify the indices where the conversation transitions into a new topic (i.e., a new event segment).

Please segment the dialogue based on **Situation Model theory**, which explains how people perceive and divide events during comprehension. Event boundaries are identified by tracking changes along the following dimensions:

1. **Goal (G)**: the primary purpose or intention of the dialogue segment  
2. **Activity (A)**: the main action or topic being discussed  
3. **Entities (E)**: the key people, objects, or concepts involved  
4. **Causality (C)**: cause–effect relations, problem–solution structures, or reasoning chains  
5. **Time (T)**: temporal references, time shifts, or changes in sequence  
6. **Space (S)**: physical location or situational context  

### Event Boundary Definition
An **event boundary** occurs when one or more of the above dimensions undergo a **substantial and sustained change**, indicating that the dialogue has entered a new situational context.
Minor elaborations, examples, or continuations of the same topic should **not** be treated as new events.

### Analysis Guidelines
- Read the dialogue utterances sequentially and evaluate their continuity with previous context across the G/A/E/C/T/S dimensions  
- If an utterance introduces a new goal, activity, or core situation, mark it as the start of a new event  
- The `boundary_index` should correspond to the **first utterance of the new event**
- In `reason`, briefly explain which dimension(s) changed (multiple dimensions may be mentioned)

### Output Format
Return **only** a JSON object with the following structure (no additional text):

```json
{
  "topic_boundaries": [
    {
      "boundary_index": <int>,
      "reason": "<brief explanation referencing the changed dimension(s)>"
    },
    {
      "boundary_index": <int>,
      "reason": "<brief explanation referencing the changed dimension(s)>"
    },
    ...
  ]
}



'''


segment_prompt_simple='''
You are given a dialogue consisting of indexed utterances.
Your task is to identify the indices where the conversation transitions into a new topic.


### Output Format
Return **only** a JSON object with the following structure (no additional text):

```json

{
  "topic_boundaries": [
    {
      "boundary_index": <int>
    },
    {
      "boundary_index": <int>
    },
    ...
  ]
}

'''

def split_chunks_from_llm_simple(chunks,prompt, tracker=None):
    """
    Split chunks based on LLM-detected topic boundaries.
    Returns a list of strings, each representing a chunk.
    """
    # 构造字典给 LLM
    chunks_dic = {n: i for n, i in enumerate(chunks)}
    
    # 调用 LLM 并解析 JSON
    response, input_tokens, output_tokens = llm_with_token_count(prompt, chunks_dic, model='gpt-4o', tracker=tracker)
    llm_data = common_use.extract_json(response)
    topic_boundaries = llm_data.get("topic_boundaries", [])
    
    # 提取边界索引并排序
    boundaries = sorted(item["boundary_index"] for item in topic_boundaries)
    
    split_chunks = []
    start_idx = 0
    
    for b in boundaries:
        # 前一个 chunk 不包含 boundary_index
        chunk_sentences = chunks[start_idx:b]
        split_chunks.append("\n".join(chunk_sentences))
        # 下一段从 boundary_index 开始
        start_idx = b
    
    # 添加最后一段
    if start_idx < len(chunks):
        split_chunks.append("\n".join(chunks[start_idx:]))

    return split_chunks


def get_split_chunks_all(chunks,prompt=situation_model_prompt):
    # 创建token tracker
    token_tracker = TokenTracker()
    
    split_chunks_all=[]
    for i in chunks:
        split_chunks=split_chunks_from_llm_simple(i,prompt, token_tracker)
        split_chunks_all.extend(split_chunks)
        print('done')
    
    # 输出token统计
    token_report = token_tracker.get_report()
    print(f"\n=== get_split_chunks_all Token消耗统计 ===")
    print(f"总输入tokens: {token_report['total_input_tokens']}")
    print(f"总输出tokens: {token_report['total_output_tokens']}")
    print(f"总tokens: {token_report['total_tokens']}")
    print(f"调用次数: {token_report['call_count']}")
    print(f"平均每次调用tokens: {token_report['avg_tokens_per_call']:.2f}")
    print("=" * 50)
    
    return split_chunks_all, token_report



#recall计算
import faiss




def search_faiss(chunks,index, query_embeddings, top_k):
    scores, indices = index.search(query_embeddings, top_k)
    # 为每个问题返回对应的chunks列表
    all_results = []
    for question_indices in indices:
        question_chunks = [chunks[i] for i in question_indices.tolist()]
        all_results.append(question_chunks)
    return all_results

def compute_evidence_score(result, retrieved_texts):
    """
    根据真实 evidence_text 与检索到的文本计算覆盖率得分

    result: list of dict, 每个 dict 必须包含 'evidence_text' (list of str)
    retrieved_texts: list of list, 每个内层列表是检索到的文本字符串

    返回: list of float，每个问题对应的得分
    """
    scores = []

    for r, retrieved in zip(result, retrieved_texts):
        true_texts = r.get('evidence_text', [])
        if not true_texts:
            scores.append(0.0)
            continue

        total = len(true_texts)
        hit_count = 0
        for t in true_texts:
            # 如果真实 evidence 文本在检索到的列表中，则算命中
            if any(t in ret for ret in retrieved):
                hit_count += 1

        scores.append(hit_count / total)

    return scores

def chunks2recall(chunks,questionlist_em,result):
    chunks_em=np.array(common_use.embedding(chunks))
    dim=1536
    index = faiss.IndexFlatIP(dim)
    index.add(chunks_em)
    for topk in [1,3,5,7,9,11]:
        retrieved_texts = search_faiss(chunks, index, questionlist_em, top_k=topk)
        scores = compute_evidence_score(result, retrieved_texts)
        print(  f"top{topk} Average Recall: {sum(scores)/len(scores)}")

def chunks2recall_return_scores(chunks, questionlist_em, result):
    chunks_em = np.array(common_use.embedding(chunks))
    dim = 1536
    index = faiss.IndexFlatIP(dim)
    index.add(chunks_em)
    
    scores_dict = {}
    for topk in [1, 3, 5, 7, 9, 11]:
        retrieved_texts = search_faiss(chunks, index, questionlist_em, top_k=topk)
        scores = compute_evidence_score(result, retrieved_texts)
        avg_score = sum(scores) / len(scores)
        scores_dict[f'top{topk}'] = avg_score
    
    return scores_dict

def generate_markdown_table(results):
    """
    生成markdown格式的结果表格
    
    Args:
        results: 包含各种chunk类型分数的字典
    
    Returns:
        str: markdown表格
    """
    # 表头
    table = "# 不同Chunk类型的召回率对比\n\n"
    table += "| Chunk Type | Top-1 | Top-3 | Top-5 | Top-7 | Top-9 | Top-11 |\n"
    table += "|------------|-------|-------|-------|-------|-------|--------|\n"
    
    # 填充数据
    for chunk_name, scores in results.items():
        row = f"| {chunk_name} | "
        row += f"{scores['top1']:.4f} | "
        row += f"{scores['top3']:.4f} | "
        row += f"{scores['top5']:.4f} | "
        row += f"{scores['top7']:.4f} | "
        row += f"{scores['top9']:.4f} | "
        row += f"{scores['top11']:.4f} |\n"
        table += row
    
    return table
def evaluate_all_chunks_and_generate_table(data_index=2):
    """
    评估所有类型的chunks并生成markdown表格
    
    Args:
        data_index: 要处理的数据索引，默认为2
    
    Returns:
        str: markdown格式的表格
    """
    # 获取数据
    result = prep.extract_q_a_evidence(data[data_index]['conversation'], data[data_index]['qa'])
    questionlist = [item['question'] for item in result]
    questionlist_em = np.array(common_use.embedding(questionlist))
    
    # 生成所有类型的chunks
    # turn_chunks = get_turn_chunks(data[data_index]['conversation'])
    # session_chunks = get_session_chunks(data[data_index]['conversation'])
    # sentence_chunks = get_sentence_chunks(data[data_index]['conversation'])
    # fixsize500 = merge_strings_with_limit(sentence_chunks, 500)
    # fixsize700 = merge_strings_with_limit(sentence_chunks, 700)
    # fixsize900 = merge_strings_with_limit(sentence_chunks, 900)
    # similarity_list_all, predicted_chunks = get_prected_chunks(sentence_chunks)
    # split_chunks = get_split_chunks_all(sentence_chunks)
    
    # 准备所有chunks类型
    chunks_types = {
        # 'turn_chunks': turn_chunks,
        # 'session_chunks': session_chunks,
        # 'sentence_chunks': sentence_chunks,
        # 'fixsize500': fixsize500,
        # 'fixsize700': fixsize700,
        # 'fixsize900': fixsize900,
        'predicted_chunks': predicted_chunks,
        # 'split_chunks': split_chunks
    }
    
    # 评估每种类型
    results = {}
    print("正在评估各种chunk类型...")
    for chunk_name, chunks in chunks_types.items():
        print(f"评估 {chunk_name}...")
        scores = chunks2recall_return_scores(chunks, questionlist_em, result)
        results[chunk_name] = scores
        print(f"完成 {chunk_name}")
    
    # 生成markdown表格
    markdown_table = generate_markdown_table(results)
    
    return markdown_table

def event_segment_main(data_index=2):
    i=data_index #选择第几个对话
    sentence_chunks=get_sentence_chunks(data[i]['conversation'])
    split_chunks, _=get_split_chunks_all(sentence_chunks)
    split_chunks_withtime=prep.get_chunks_with_timestamps(split_chunks,data[i]['conversation'])

    return split_chunks_withtime

def event_segment_simple(data_index=2):
    i=data_index #选择第几个对话
    sentence_chunks=get_sentence_chunks(data[i]['conversation'])
    split_chunks, _=get_split_chunks_all(sentence_chunks,segment_prompt_simple)
    split_chunks_withtime=prep.get_chunks_with_timestamps(split_chunks,data[i]['conversation'])

    return split_chunks_withtime



#主程序运行部分
# if __name__ == "__main__":
    # i=1 #选择第几个对话
    # turn_chunks=get_turn_chunks(data[i]['conversation'])
    # session_chunks=get_session_chunks(data[i]['conversation'])
    # sentence_chunks_flat=get_sentence_chunks_flat(data[i]['conversation'])
    # sentence_chunks=get_sentence_chunks(data[i]['conversation'])

    # fixsize500=merge_strings_with_limit(sentence_chunks,500)
    # fixsize700=merge_strings_with_limit(sentence_chunks,700)
    # fixsize900=merge_strings_with_limit(sentence_chunks,900)
    # similarity_list_all, predicted_chunks = get_prected_chunks(sentence_chunks)
    # split_chunks=get_split_chunks_all(sentence_chunks)

    # #获得recall分数表格
    # table = evaluate_all_chunks_and_generate_table(data_index=i)



    # #获得带时间戳的chunks
    # split_chunks_withtime=prep.get_chunks_with_timestamps(split_chunks,data[i]['conversation'])

#%%探索性


def chunks2score(chunks):
    chunks_em=np.array(common_use.embedding(chunks))
    score=np.dot(chunks_em, chunks_em.T)

    return score


def mean_without_diagonal(mat: np.ndarray):
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return mat[mask].mean()

import matplotlib.pyplot as plt
import numpy as np

def plot_similarity_heatmap(
    sim_matrix,
    labels=None,
    title="Similarity Heatmap",
    cmap="viridis",
    show_values=False
):
    """
    sim_matrix: 2D array-like (N x N)
    labels: 行/列标签（list of str），可选
    show_values: 是否在格子里显示数值
    """

    sim_matrix = np.array(sim_matrix)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(sim_matrix, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)

    if show_values:
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                plt.text(
                    j, i,
                    f"{sim_matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if sim_matrix[i, j] > 0.6 else "black"
                )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_list(data, title="Data Visualization"):
    """
    data: List[float]
    """
   
    plt.figure()
    plt.plot(data)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(title)
    plt.show()