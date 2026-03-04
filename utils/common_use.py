
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import re
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import math
from concurrent.futures import ThreadPoolExecutor,as_completed

from typing import Iterable, Tuple, Union, List, Any, Dict
import pickle
# 读取 .env 文件
load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def llm(sys,context,model="qwen2.5-7b-instruct"):
    response = client.chat.completions.create(  #
                model=model,  # 填写需要调用的模型编码
                messages=[
                    {"role": "system", "content":str(sys)},
                    {"role": "user", "content":str(context)},
                ]).choices[0].message.content
    return response

def llm_t0(sys,context,model="qwen2.5-7b-instruct"):
    response = client.chat.completions.create(  #
                model=model,  # 填写需要调用的模型编码
                temperature=0,
                messages=[
                    {"role": "system", "content":str(sys)},
                    {"role": "user", "content":str(context)},
                ]).choices[0].message.content
    return response

def llm_t0_parallel(
    sys,
    context_list,
    model="qwen2.5-7b-instruct",
    max_workers=8
    ):
    results = ['error'] * len(context_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(llm_t0, sys, ctx, model): i
            for i, ctx in enumerate(context_list)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[llm_t0_parallel ERROR] index={idx}, error={e}")
                results[idx] = 'error'

    return results
def count_tokens_gpt(text):
    import tiktoken

    # 选择模型编码器（和你将要用的模型一致）
    encoding = tiktoken.encoding_for_model("gpt-4")  # 或 "gpt-3.5-turbo"

    tokens = encoding.encode(text)
    #print("Token 数:", len(tokens))

    return len(tokens)

def count_tokens(texts: List[str]) -> int:
    """计算文本列表的总tokens数（简化估算）"""
    total = 0
    for text in texts:
        # 简化估算：假设每个字符对应0.74 tokens（实际应使用tiktoken）
        total += math.ceil(len(text) * 0.74)
    return total

def split_texts(texts: List[str], max_tokens: int = 8000) -> List[List[str]]:
    """将文本列表拆分为不超过max_tokens的批次"""
    if not texts:
        return [[]]
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        text_tokens = math.ceil(len(text) * 0.74)  # 简化估算
        
        # 如果当前批次添加该文本后会超过限制，则新建批次
        if current_tokens + text_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        
        current_batch.append(text)
        current_tokens += text_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def embedding(context,model="text-embedding-3-small"):
    """生成文本的嵌入向量，支持大输入分批处理，空值返回0向量"""
    # 处理空输入
    if not context:
        print("接收到空输入，返回空列表。")
        return []
    
    # 确保输入是列表
    if not isinstance(context, list):
        print(f"输入不是列表，已自动转换为列表，长度为: {len(context)}")
        context = [context]
    
    # 记录原始索引和有效文本
    valid_texts = []
    valid_indices = []
    original_length = len(context)
    
    for idx, item in enumerate(context):
        # 检查是否为有效文本
        text = str(item).strip() if item is not None else ""
        if text:
            valid_texts.append(text)
            valid_indices.append(idx)
    
    # 初始化结果列表为全0向量
    # 假设嵌入向量维度为1536（text-embedding-3-small的默认维度）
    embedding_dim = 1536
    result = [[0.0 for _ in range(embedding_dim)] for _ in range(original_length)]
    
    if not valid_texts:
        return result  # 全部为无效值，返回全0向量列表
    
    # 计算总tokens并分批
    total_tokens = count_tokens(valid_texts)
    all_embeddings = []
    
    if total_tokens <= 8000:

        # 直接处理
        response = client.embeddings.create(
            input=valid_texts,
            #model="text-embedding-3-small"
            model=model
        )
        all_embeddings = [item.embedding for item in response.data]

    else:
        # 分批处理
        batches = split_texts(valid_texts)
        print("\n文本总长度超出单次请求限制，将进行分批处理。")
        print(f"共分成 {len(batches)} 个批次。")
        for i,batch in enumerate(batches):
            print(f"\n正在处理批次 {i+1}/{len(batches)} ")
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            all_embeddings.extend([item.embedding for item in response.data])
    
    # 将有效嵌入向量放回正确位置
    for idx, embedding in zip(valid_indices, all_embeddings):
        result[idx] = embedding
    
    return result
def extract_json(text: str) -> dict:
    """
    从文本中提取JSON对象，支持以下情况：
    1. 纯JSON格式的文本
    2. 被```json和```包裹的JSON代码块
    3. 文本中包含的单个JSON对象
    """
    # 情况1：检查是否为被```json包裹的代码块
    if "```json" in text.lower() and "```" in text:
        # 提取被```json和```包围的内容
        code_block_pattern = r'```json(.*?)```'
        match = re.search(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass  # 继续尝试其他方法
    
    # 情况2：检查是否为普通JSON代码块（被```包裹）
    if "```" in text:
        code_block_pattern = r'```(.*?)```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass  # 继续尝试其他方法
    
    # 情况3：尝试直接解析整个文本
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # 继续尝试其他方法
    
    # 情况4：从文本中提取JSON对象
    json_pattern = r'(\{.*?\})'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # 无法解析提取的内容
    
   
    # 未找到有效JSON
    return None

def save_class(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)  # 序列化并保存
    print(f"类实例已保存到 {file_path}")

# 从本地文件加载类实例
def load_class(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)  # 反序列化
    print(f"已从 {file_path} 加载类实例")
    return obj

def save_to_json(data: Any, file_path: str) -> bool:
    """
    将数据保存为JSON格式文件
    
    参数:
        data: 要保存的数据，可以是字典、列表等可序列化对象
        file_path: 保存的文件路径
    
    返回:
        保存成功返回True，失败返回False
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            # 确保中文等特殊字符正确保存
            json.dump(data, file, ensure_ascii=False, indent=4)
        return True
    except TypeError as e:
        print(f"数据无法序列化为JSON: {e}")
    except IOError as e:
        print(f"文件写入错误: {e}")
    except Exception as e:
        print(f"保存JSON时发生错误: {e}")
    return False

def load_from_json(file_path: str) -> Any:
    """
    从JSON格式文件导入数据
    
    参数:
        file_path: 要导入的JSON文件路径
    
    返回:
        成功返回解析后的数据，失败返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except IOError as e:
        print(f"文件读取错误: {e}")
    except Exception as e:
        print(f"读取JSON时发生错误: {e}")
    return None


