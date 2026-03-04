from datasets import load_dataset
import tiktoken
import re


def extract_questions(text_list):
    """
    从文本列表中提取 'Now Answer the Question:' 后面的纯问题文本
    
    参数：
        text_list (list of str): 文本列表，每个元素是一个包含问题和选项的字符串
    
    返回：
        questions (list of str): 提取出的纯问题文本列表
    """
    questions = []
    pattern = r"Now Answer the Question:\s*(.*?)\n[A-D]\."  # 假设选项都是 A. B. C. D.
    
    for text in text_list:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            question = match.group(1).strip()
            questions.append(question)
        else:
            questions.append(None)  # 如果匹配失败，可返回 None 或空字符串
    
    return questions


def split_text_by_tokens_preserve_sentences(text, max_tokens=512, model="gpt-4o-mini"):
    """
    将文本切分成不超过 max_tokens 的 chunk，并尽量保证句子完整性。

    参数:
        text (str): 需要被切分的原始文本。
        max_tokens (int): 每个 chunk 的最大 token 数量。
        model (str): 用于选择 tiktoken 编码的模型名称。

    返回:
        list: 一个包含所有切分后文本块 (chunk) 的列表。
    """
    if not text:
        return []

    # 使用正则表达式进行句子分割，这是一个简化的分句逻辑
    # 它会在 '.', '?', '!' 后面且跟着换行或大写字母的地方进行分割
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    
    # 如果没有找到符合条件的句子结尾，就用换行符分割
    if len(sentences) == 1:
        sentences = text.split('\n')
        
    # 如果仍然只有一个元素，就按段落分割（两个及以上换行符）
    if len(sentences) == 1:
        sentences = re.split(r'\n{2,}', text)
        
    encoding = tiktoken.encoding_for_model(model)
    
    chunks = []
    current_chunk_sentences = []
    current_chunk_token_count = 0

    for sentence in sentences:
        # 计算当前句子的 token 长度
        sentence_tokens = encoding.encode(sentence)
        sentence_token_count = len(sentence_tokens)

        # 如果单个句子就超过了 max_tokens，我们不得不将其单独作为一个 chunk
        if sentence_token_count > max_tokens:
            if current_chunk_sentences:
                # 先保存当前正在构建的 chunk
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_chunk_token_count = 0
            
            # 将这个超长句子单独加入 chunks
            chunks.append(sentence)
            continue

        # 检查如果将这个句子添加到当前 chunk，是否会超出 token 限制
        if current_chunk_token_count + sentence_token_count <= max_tokens:
            current_chunk_sentences.append(sentence)
            current_chunk_token_count += sentence_token_count
        else:
            # 如果会超出，就先保存当前 chunk
            chunks.append(" ".join(current_chunk_sentences))
            
            # 然后开始一个新的 chunk，并将当前句子加入
            current_chunk_sentences = [sentence]
            current_chunk_token_count = sentence_token_count

    # 添加最后一个未完成的 chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks
def split_list_by_tokens(input_list, max_tokens=512, model="gpt-4o-mini"):
    """
    注意：输入是一个列表list，list中每一个值为str
    将一个字符串列表切分成多个文本块 (chunk)，每个 chunk 的 token 数量不超过 max_tokens。
    列表中的每个元素都将被视为一个不可分割的整体。

    参数:
        input_list (list): 需要被切分的字符串列表。列表中的每个元素是一个不可分割的块。
        max_tokens (int): 每个 chunk 的最大 token 数量。
        model (str): 用于选择 tiktoken 编码的模型名称。

    返回:
        list: 一个包含所有切分后文本块 (chunk) 的列表。
    """
    if not input_list:
        return []

    # 过滤掉列表中的空字符串，避免不必要的处理
    non_empty_items = [item for item in input_list if item.strip()]
    if not non_empty_items:
        return []
        
    encoding = tiktoken.encoding_for_model(model)
    
    chunks = []
    current_chunk_items = []
    current_chunk_token_count = 0

    for item in non_empty_items:
        # 计算当前列表元素的 token 长度
        # 注意：这里我们计算的是元素本身的 token 数，不包含将来拼接时添加的空格
        item_tokens = encoding.encode(item)
        item_token_count = len(item_tokens)

        # --- 核心逻辑 ---
        # 1. 如果单个元素就超过了 max_tokens，这是一个问题。
        #    我们不得不将其单独作为一个 chunk，即使它超出了限制。
        #    这是为了保证元素的完整性。
        if item_token_count > max_tokens:
            # 如果当前 chunk 不为空，先保存它
            if current_chunk_items:
                chunks.append(" ".join(current_chunk_items))
                current_chunk_items = []
                current_chunk_token_count = 0
            
            # 将这个超长元素单独加入 chunks
            chunks.append(item)
            continue

        # 2. 检查添加当前元素后，当前 chunk 是否会超出限制。
        #    这里要考虑到，当我们用 " ".join() 拼接时，会为每个元素（除了第一个）
        #    增加一个空格，而一个空格在大多数编码中占 1 个 token。
        #    所以，新的总 token 数 = 当前总数 + 空格数(即当前 chunk 中已有元素的数量) + 新元素的 token 数
        additional_tokens = item_token_count + (len(current_chunk_items) > 0)
        
        if current_chunk_token_count + additional_tokens <= max_tokens:
            # 如果不会超出，就将元素添加到当前 chunk
            current_chunk_items.append(item)
            current_chunk_token_count += additional_tokens
        else:
            # 如果会超出，就先保存当前 chunk
            chunks.append(" ".join(current_chunk_items))
            
            # 然后开始一个新的 chunk，并将当前元素加入
            current_chunk_items = [item]
            current_chunk_token_count = item_token_count

    # 添加最后一个未完成的 chunk
    if current_chunk_items:
        chunks.append(" ".join(current_chunk_items))

    return chunks



def preprocess_sessions_with_rounds(raw, rounds=1):
    result = []

    # 每轮对话等于 2 句（比如 rounds=2 → group_size=4）
    group_size = rounds * 2

    for key, value in raw.items():
        if key.startswith("session_") and key.endswith("_date_time"):
            session_name = key.replace("_date_time", "")
            date_time = value
            dialogues = raw.get(session_name, [])

            paired = []
            temp_group = []

            for item in dialogues:
                speaker = item["speaker"]
                text = item["text"]
                # 跳过空内容
                if not text or not text.strip():
                    continue
                temp_group.append(f"\n{speaker}: {text}")

                # 达到 group_size → 打包一个 group
                if len(temp_group) == group_size:
                    paired.append([date_time] + temp_group)
                    temp_group = []

            # 如果最后不足 group_size → 合并到上一组
            if len(temp_group) > 0:
                if paired:
                    paired[-1].extend(temp_group)
                else:
                    # 整个 session 没有任何完整 group
                    paired.append([date_time] + temp_group)

            result.extend(paired)

    return result


def preprocess_sessions_with_rounds_no_timestamp(raw, rounds=1):
    """
    处理会话数据，将对话按轮数分组，不添加时间戳，并按session分组返回。

    参数：
        raw (dict): 原始数据字典，包含会话信息
        rounds (int): 每组的轮数，默认为1（每轮等于2句话）

    返回：
        list: 嵌套列表结构，外层列表代表不同session，内层列表包含该session的对话组
    """
    result = []

    # 每轮对话等于 2 句（比如 rounds=2 → group_size=4）
    group_size = rounds * 2

    for key, value in raw.items():
        if key.startswith("session_") and key.endswith("_date_time"):
            session_name = key.replace("_date_time", "")
            dialogues = raw.get(session_name, [])

            paired = []
            temp_group = []

            for item in dialogues:
                speaker = item["speaker"]
                text = item["text"]
                # 跳过空内容
                if not text or not text.strip():
                    continue
                temp_group.append(f"{speaker}: {text}")

                # 达到 group_size → 打包一个 group
                if len(temp_group) == group_size:
                    paired.append("\n".join(temp_group))
                    temp_group = []

            # 如果最后不足 group_size → 合并到上一组
            if len(temp_group) > 0:
                if paired:
                    paired[-1] = paired[-1] + "\n".join(temp_group)
                else:
                    # 整个 session 没有任何完整 group
                    paired.append("\n".join(temp_group))

            # 将每个session的结果作为单独的列表添加到result中
            if paired:  # 只有当该session有对话内容时才添加
                result.append(paired)

    return result

def preprocess_dialogue_to_single(data):
    """
    将多 session 对话数据转换成嵌套列表，每个 session 是一个列表，
    每条对话格式为 'speaker: text'，如果有图像则附加 ',image:query'
    
    Args:
        data (dict): 原始对话字典

    Returns:
        List[List[str]]: 处理后的嵌套列表
    """
    sessions = []
    
    # 遍历所有 key 找到 session
    for key in data:
        if key.startswith("session_") and not key.endswith("_date_time"):
            session_dialogues = []
            for turn in data[key]:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "").strip()
                # 如果有图像，则加上 query
                if "img_url" in turn and "query" in turn:
                    text += f",image:{turn['query']}"
                session_dialogues.append(f"{speaker}:{text}")
            sessions.append(session_dialogues)
    
    return sessions

def extract_q_a_evidence(conversation, questions):
    result = []

    # 建立 dia_id -> text 的索引
    dia_dict = {}
    for key, session in conversation.items():
        if key.startswith('session_') and not key.endswith('_date_time'):
            for entry in session:
                dia_dict[entry['dia_id']] = entry['text']

    # 遍历问题列表
    for q in questions:
        if 'answer' not in q or not q['answer']:
            continue
        if 'evidence' not in q or not q['evidence']:
            continue
        
        # 查找 evidence 对应的文本
        evidence_texts = [dia_dict[eid] for eid in q['evidence'] if eid in dia_dict]
        if not evidence_texts:
            continue
        
        result.append({
            'question': q['question'],
            'answer': q['answer'],
            'evidence_text': evidence_texts
        })
    
    return result

import re
from typing import List


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



import re
from typing import List, Dict, Any

def strip_speaker(line: str) -> str:
    """去掉行首 speaker 前缀"""
    return re.sub(r'^[A-Za-z_ ]+:\s*', '', line).strip()

def remove_image_content(line: str) -> str:
    """
    删除句子中任何 ',image:...' 或 'image:...' 内容，保留其他文本
    """
    return re.sub(r',?image:[^,\n]*', '', line, flags=re.IGNORECASE).strip()

def get_chunks_with_timestamps(chunks: List[str], conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    使用 chunk 第一行作为锚点匹配 session，删除 image 内容
    返回 [{"chunk": chunk, "timestamp": timestamp}, ...]
    """
    # 构建 session 文本 + 时间戳列表
    sessions = [
        ("\n".join(turn["text"].strip() for turn in conversation.get(k.replace("_date_time",""), []) if "text" in turn), v)
        for k, v in conversation.items() if k.startswith("session_") and k.endswith("_date_time")
    ]

    results = []
    for chunk in chunks:
        # 取非空行
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        if not lines:
            timestamp = None
        else:
            # 取第一行，去掉 speaker 和 image 内容
            anchor = strip_speaker(remove_image_content(lines[0]))
            # 在 session 文本中匹配
            timestamp = next((ts for text, ts in sessions if anchor in text), None)

        results.append({"chunk": chunk, "timestamp": timestamp})

    return results



