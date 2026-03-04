import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
from utils import common_use 
from core import prompt

import json 
def get_questions(n):
    file_path='data/locomo10.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questionlist=[]
    right_answer=[]
    questionlistlabel=[]
    for ii in data[n]['qa']:
        if 'answer' in ii:
            questionlist.append(ii['question'])
            right_answer.append(ii['answer'])
            questionlistlabel.append(ii['category'])

    return questionlist,right_answer,questionlistlabel




def abstract_questions_answer(evidencechunks,questionlist,model='gpt-4o'):
    llmmodel=model
    llmanswerlist=[]
    for i in range(len(questionlist)):   
        try:
            llm_answer =(common_use.extract_json(common_use.llm_t0(prompt.answerprompt_abstract, 
                f'Documents:{evidencechunks[i]},Question:{questionlist[i]}', 
                model=llmmodel))['answer'])
            llmanswerlist.append(llm_answer)
        except:
            llmanswerlist.append('error')
            print('error')
    return llmanswerlist


#long context回答问题
def long_context_answer(evidencechunk,questionlist,model='gpt-4o'):
    llmmodel=model
    llmanswerlist_long=[]

    for i in questionlist:   
        llm_answer =(common_use.extract_json(common_use.llm_t0(prompt.answerprompt_abstract, 
            f'Documents:{evidencechunk},Question:{i}', 
            model=llmmodel))['answer'])
        llmanswerlist_long.append(llm_answer)
    return llmanswerlist_long

def specific_questions_answer(evidencechunks,questionlist,model='gpt-4o'):
    llmmodel=model
    llmanswerlist=[]
    for i in range(len(questionlist)):   
        try:
            llm_answer =(common_use.extract_json(common_use.llm_t0(prompt.answerprompt_withoutvague, 
                f'Documents:{evidencechunks[i]},Question:{questionlist[i]}', 
                model=llmmodel))['answer'])
            llmanswerlist.append(llm_answer)
        except:
            llmanswerlist.append('error')
            print('error')
    return llmanswerlist

def specific_questions_answer_parallel( 
    evidencechunks,
    questionlist,
    model='gpt-4o',
    max_workers=8
):
    assert len(evidencechunks) == len(questionlist)

    # 1️⃣ 构造并行调用的 context 列表
    context_list = [
        f'Documents:{evidencechunks[i]},Question:{questionlist[i]}'
        for i in range(len(questionlist))
    ]

    # 2️⃣ 并行调用 LLM（system prompt 不变）
    raw_responses = common_use.llm_t0_parallel(
        prompt.answerprompt_withoutvague,
        context_list,
        model=model,
        max_workers=max_workers
    )

    # 3️⃣ 后处理（JSON 抽取 + 兜底）
    llmanswerlist = []
    for i, resp in enumerate(raw_responses):
        try:
            if resp == 'error':
                raise ValueError("llm call failed")

            answer = common_use.extract_json(resp)['answer']
            llmanswerlist.append(answer)
        except Exception as e:
            print(f"[specific_questions_answer ERROR] idx={i}, error={e}")
            llmanswerlist.append('error')

    return llmanswerlist




def LLMfilter(questionlist,evidencechunks,model='gpt-4o',max_workers=10):
    context_list = [prompt.filter_prompt.format(retrieved_texts=evidencechunks[i],question=questionlist[i]) for i in range(len(questionlist))]
    raw_responses = common_use.llm_t0_parallel(
        sys='',
        context_list=context_list,
        model=model,
        max_workers=max_workers
    )

    filtered_chunks = []
    for n,resp in enumerate(raw_responses):
        try:
            if resp == 'error':
                raise ValueError("llm call failed")

            filtered_text = common_use.extract_json(resp)['filtered_text']
            filtered_chunks.append(filtered_text)
        except Exception as e:
            print(f"[LLMfilter ERROR] error={e}")
            filtered_chunks.append(evidencechunks[n])

    return filtered_chunks


def rateLLM(questionlist, right_answerlist, answerlist,type):
    if type=='0-1':
        judgeprompt=prompt.judgeprompt
    if type=='0-100':
        judgeprompt=prompt.judgeprompt_1_100
    rate_all=[]
    for i in range(len(questionlist)):
        try:
            rate=common_use.extract_json(common_use.llm(judgeprompt,f'User Question: {questionlist[i]}\nRight Answer: {right_answerlist[i]}\nBot Response {answerlist[i]}"'))['rating']
            rate_all.append(rate)
        except:
            print('error')
            rate_all.append('error')
    return rate_all

def rateLLM_parallel(
    questionlist,
    right_answerlist,
    answerlist,
    type,
    model='gpt-4o',
    max_workers=8
):
    assert len(questionlist) == len(right_answerlist) == len(answerlist)

    # 1️⃣ 选择 judge prompt
    if type == '0-1':
        judgeprompt = prompt.judgeprompt
    elif type == '0-100':
        judgeprompt = prompt.judgeprompt_1_100
    else:
        raise ValueError(f"Unknown type: {type}")

    # 2️⃣ 构造并行 context 列表
    context_list = [
        (
            f"User Question: {questionlist[i]}\n"
            f"Right Answer: {right_answerlist[i]}\n"
            f"Bot Response: {answerlist[i]}"
        )
        for i in range(len(questionlist))
    ]

    # 3️⃣ 并行调用 LLM
    raw_responses = common_use.llm_t0_parallel(
        judgeprompt,
        context_list,
        model=model,
        max_workers=max_workers
    )

    # 4️⃣ 解析评分结果
    rate_all = []
    for i, resp in enumerate(raw_responses):
        try:
            if resp == 'error':
                raise ValueError("llm call failed")

            rate = common_use.extract_json(resp)['rating']
            rate_all.append(rate)
        except Exception as e:
            print(f"[rateLLM ERROR] idx={i}, error={e}")
            rate_all.append('error')

    return rate_all
from collections import defaultdict

def average_score_by_label(scores, labels):
    """
    根据 label 计算各类别的平均分

    Args:
        scores (list): 分数列表（如 int / float）
        labels (list): label 列表（与 scores 等长）
        category_map (dict): label -> 类别名称 的映射

    Returns:
        dict: {类别名称: 平均分}
    """
    assert len(scores) == len(labels), "scores 和 labels 长度必须一致"
    category_map = {
        4: 'Single-hop（单跳）',
        1: 'Multi-hop（多跳）',
        2: 'Temporal（时间推理）',
        3: 'Open-domain（开放域知识）'
    }
    score_sum = defaultdict(float)
    count = defaultdict(int)

    for score, label in zip(scores, labels):
        if score == 'error' or score is None:
            continue

        category = category_map.get(label, 'Unknown')
        score_sum[category] += score
        count[category] += 1

    avg_scores = {
        category: score_sum[category] / count[category]
        for category in score_sum
        if count[category] > 0
    }

    return avg_scores

def get_answer_and_rate(context,i):
    questionlist,right_answer,questionlistlabel=get_questions(i)

    llmanswerlist=specific_questions_answer_parallel(context,questionlist,model='gpt-4o',max_workers=100)


    rate_all=rateLLM_parallel(questionlist,right_answer,llmanswerlist,'0-100',model='gpt-4o',max_workers=100)
    avg_scores=average_score_by_label(rate_all,questionlistlabel)
    print('各类别平均分：',avg_scores)
    overall_avg = (
    sum(avg_scores.values()) / len(avg_scores)
    if avg_scores else None)

    print('按类平均分：', overall_avg)
    # rate_all=rate.rateLLM(questionlist,right_answer,llmanswerlist,'0-100')
    print('所有平均分',sum(rate_all)/len(rate_all))
    return llmanswerlist,rate_all
