import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import utils.common_use as common_use
import numpy as np
import json
import networkx as nx
from typing import List, Dict


def embeding_retival(chunks,chunks_em,questionlist_em,topK):
    context_chunks_all=[]
    for i in questionlist_em:
        # 向量已归一化，点乘即余弦相似度
        cosine_scores = chunks_em @ i.T   # shape (n,)

        # topk 索引
        topk_indices = np.argsort(cosine_scores)[-topK:][::-1]  # 从大到小
        # 对应文本块
        topk_indices=sorted(topk_indices)
        evidence = [chunks[n] for n in topk_indices]
        context_chunks_all.append(evidence)
        #计算一下上下文的平均token数
    # token_num_list=[]
    # for i in context_chunks_all:
    #     token_num=common_use.count_tokens(str(i))
    #     #print(len(i),token_num)
    #     token_num_list.append(token_num)
    #print('平均token数：',sum(token_num_list)/len(token_num_list))
    return context_chunks_all



def get_questions(n):
    file_path='data/locomo10.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questionlist=[]
    right_answer=[]
    for ii in data[n]['qa']:
        if 'answer' in ii:
            questionlist.append(ii['question'])
            right_answer.append(ii['answer'])
            # questionlistlabel.append(ii['category'])

    return questionlist,right_answer

def topk_retival(chunks,topK,i):
    chunks_em=np.array(common_use.embedding(chunks))
    questionlist,right_answer=get_questions(i)
    questionlist_em=np.array(common_use.embedding(questionlist))
    context_chunks_all=embeding_retival(chunks,chunks_em,questionlist_em,topK)
    return context_chunks_all



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def ppr_with_question(
    chunks_em: List[np.ndarray],
    questionlist_em: List[np.ndarray],
    G: nx.Graph,
    topk: int,
    top_sim_k: int = 3,
    alpha: float = 0.85,
) -> List[List[int]]:
    """
    对每个 question 执行：
    1. 与 chunks_em 计算余弦相似度
    2. 取相似度最高的 top_sim_k 个 chunk
    3. 以其为 personalization，执行 PPR
    4. 返回 PageRank 值最高的 topk 个节点索引
    """

    results = []

    chunks_em = [np.asarray(v) for v in chunks_em]
    questionlist_em = [np.asarray(v) for v in questionlist_em]

    num_chunks = len(chunks_em)

    for q_idx, q_em in enumerate(questionlist_em):
        # ---------- 1. 计算余弦相似度 ----------
        sims = np.zeros(num_chunks)
        for i, c_em in enumerate(chunks_em):
            sims[i] = cosine_similarity(q_em, c_em)

        # ---------- 2. 取相似度最高的 top_sim_k 个 ----------
        top_sim_indices = np.argsort(sims)[-top_sim_k:][::-1]

        # ---------- 3. 构造 personalization ----------
        personalization: Dict[int, float] = {node: 0.0 for node in G.nodes()}
        for idx in top_sim_indices:
            if idx in personalization:
                personalization[idx] = 1.0

        # ---------- 4. 执行 Personalized PageRank ----------
        pr = nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
        )

        # ---------- 5. 取 topk ----------
        topk_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:topk]
        topk_indices = [node for node, _ in topk_nodes]

        results.append(topk_indices)

    return results

def ppr_retival(chunks,graph,topk,i):
    chunks_em=np.array(common_use.embedding(chunks))
    questionlist,right_answer=get_questions(i)
    questionlist_em=np.array(common_use.embedding(questionlist))
    topk_indices_all=ppr_with_question(chunks_em,questionlist_em,graph,topk)
    
    context_chunks_all=[]
    for indices in topk_indices_all:
        evidence = [chunks[n] for n in indices]
        context_chunks_all.append(evidence)
    return context_chunks_all
