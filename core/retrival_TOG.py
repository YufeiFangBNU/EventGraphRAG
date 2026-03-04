#%%
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import utils.common_use as common_use
import json
import numpy as np

#加载数据



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





#起始节点的选择


def chornode(chunks,questions,top_k):

    questions_em=np.array(common_use.embedding(questions))
    chunks_em=np.array(common_use.embedding(chunks))

    import faiss
    dim=1536
    index = faiss.IndexFlatIP(dim)
    index.add(chunks_em)

    scores, indices = index.search(questions_em, top_k)
    return indices.tolist()



#TOG
relation_prompt = """
You are an expert at discovering semantic relevance and conceptual connections in text.

You are given a question and a set of numbered documents.
Each document is represented ONLY by its summary, not the full text.

Your task is to evaluate the *potential relevance* of each document summary for answering the question,
focusing on the overall topic, concept, or thematic alignment with the question.
Do NOT focus on whether the summary contains fine details or explicit answers; judge based on general relevance and conceptual direction.

Scoring scale (integer from 1 to 10), based on **potential relevance inferred from the summary**:
- 1: Clearly irrelevant; the summary is unrelated in topic or concept to the question
- 3: Very weak potential relevance; only vague or incidental overlap in theme
- 5: Moderate potential relevance; the summary indicates related concepts or context, but usefulness is uncertain
- 7: High potential relevance; the summary is closely related in topic or concept and likely useful
- 9–10: Very high potential relevance; the summary is strongly aligned with the question's theme and likely central to answering it

【Output format specification】
Strictly output a SINGLE valid JSON object.
Output JSON only. No extra text.

The JSON structure must be:
{
  "results": [
    {
      "doc_id": <document ID>,
      "score": <integer from 1 to 10>,
      "explanation": "<1–2 sentences explaining the POTENTIAL relevance based on the general topic or concept of the document summary>"
    },
    ...
  ]
}
"""


suficient_judge_prompt='''
You are a rigorous knowledge evaluation assistant.
You will be given a 【Question】 and a set of 【Relevant Documents】.
Your task is to determine whether the documents alone are sufficient to fully and accurately answer the question.

Judgment criteria:
- 0 (Insufficient):
  The documents lack the key information required to answer the question, or have very low relevance.
  A grounded answer cannot be produced based on the documents.

- 1 (Partially sufficient but possibly incomplete):
  The documents contain some information related to the question and can address certain aspects,
  but important details are missing, incomplete, or uncertain.

- 2 (Fully sufficient):
  The documents contain all key information required to answer the question completely and clearly.
  No external knowledge, assumptions, or additional sources are needed.

Output requirements (very important):
- Output ONLY a single JSON object
- The JSON object must contain exactly two fields:
  - "result": must be one of 0, 1, or 2
  - "justification": a brief explanation of the judgment
- Justification rules:
  - 1–3 sentences only
  - Describe only whether the documents cover the information needed to answer the question
  - Do NOT provide step-by-step reasoning or detailed analysis
- Do NOT output any extra text, explanations, or formatting

Output example:
{
  "result": 1,
  "justification": "The documents explain the core concept related to the question but omit key conditions and full details required for a complete answer."
}



'''


class ToG:
    def __init__(self, G, question,chor_node_id):
        self.G = G
        self.question = question
        
        self.chor_node_id = list(chor_node_id)
        self.node_history = set(self.chor_node_id)  # 已访问节点
        self.relevant_docs = []
        for node_id in self.chor_node_id:
            content = self.G.nodes[node_id].get('content')
            timestamp = self.G.nodes[node_id].get('timestamp', '')
            if content:
                doc_entry = f"time:{timestamp},content:{content}"
                self.relevant_docs.append(doc_entry)
    def suficient_judge(self):
        context = (
            f"【Question】\n{self.question}\n\n"
            f"【Relevant Documents】\n" + "\n".join(self.relevant_docs)
        )
        response = common_use.llm_t0(suficient_judge_prompt, context,model='gpt-4o')
        try:
            result = common_use.extract_json(response)
            return result.get('result', 'error')
        except Exception as e:
            print(f"[suficient_judge] JSON error: {e}")
            return 'error'

    def llm_find(self):
        candidate_neighbors = set()

        for node_id in self.chor_node_id:
            for neighbor in self.G.neighbors(node_id):
                if neighbor not in self.node_history:
                    candidate_neighbors.add(neighbor)
                    # ⭐ 关键：一旦进入候选集，就标记为已访问
                    self.node_history.add(neighbor)

        if not candidate_neighbors:
            return 'no_more_nodes'

        neighbor_summaries = {
            nid: self.G.nodes[nid].get('summary', '')
            for nid in candidate_neighbors
        }

        context = (
            f"【Question】\n{self.question}\n\n"
            f"【Relevant Documents Summaries】\n{neighbor_summaries}"
        )

        response = common_use.llm_t0(relation_prompt, context,model='gpt-4o')

        try:
            return common_use.extract_json(response)
        except Exception as e:
            print(f"[llm_find] JSON error: {e}")
            return 'error'

    def main(self, max_iter=3, score_threshold=5):
        """
        主循环：
        - sufificient_judge == 2 → 'suficient'
        - 没有邻居节点 → 'no_more_nodes'
        - 没有高分节点 → 'no_more_nodes'
        - 达到最大迭代次数 → 'max_iter_reached'
        - LLM / JSON 错误 → 'error'
        """
        for step in range(max_iter):
            print(f"\n[ToG] Step {step + 1}")

            # 1️⃣ 判断是否已充分回答
            suficient_result = self.suficient_judge()
            print("[ToG] suficient_judge:", suficient_result)

            if suficient_result == 'error':
                return 'error'
            if suficient_result == 2:
                return 'suficient'

            # 2️⃣ 查找新节点
            relation_result = self.llm_find()
            if relation_result == 'no_more_nodes':
                return 'no_more_nodes'
            if relation_result == 'error':
                return 'error'
            print('[ToG] llm_find results:', relation_result)
            # 3️⃣ 筛选高分节点
            new_nodes = [
                item['doc_id']
                for item in relation_result.get('results', [])
                if item.get('score', 0) >= score_threshold
            ]
            print(f"[ToG] New high-score nodes: {new_nodes}")
            if not new_nodes:
                return 'no_more_nodes'  # 有邻居但没有高分节点

            # 4️⃣ 更新状态
            self.chor_node_id = new_nodes
            for node_id in new_nodes:
                content = self.G.nodes[node_id].get('content')
                timestamp = self.G.nodes[node_id].get('timestamp', '')
                if content:
                    doc_entry = f"time:{timestamp},content:{content}"
                    self.relevant_docs.append(doc_entry)

        # 达到最大循环次数
        return 'max_iter_reached'

#%%执行
if __name__ == "__main__":
    chunks=common_use.load_from_json('session1_splitchunks.json')
    graph=common_use.load_class('session1_graph_withtime.pkl')
    questionlist,right_answer=get_questions(1)
    chor_node_id=chornode(chunks,questionlist,3)


#串行
# retrival_results=[]
# for i in range(len(questionlist)):
#     tog=ToG(graph,questionlist[i],chor_node_id[i])
#     tog.main()
#     related_docs=tog.relevant_docs
#     retrival_results.append(related_docs)
def TOG_main(chunks,graph,i):
    questionlist,right_answer=get_questions(i)

    chor_node_id=chornode(chunks,questionlist,3)
    #并行
    from concurrent.futures import ThreadPoolExecutor

    # 定义单个问题的处理函数
    def process_question(idx):
        tog = ToG(graph, questionlist[idx], chor_node_id[idx])
        tog.main()  # 调用模型 API
        return tog.relevant_docs

    # 并行执行
    with ThreadPoolExecutor(max_workers=20) as executor:  # 可以根据你的API并发限制调整线程数
        retrival_results = list(executor.map(process_question, range(len(questionlist))))
    return retrival_results
# retrival_results[i] 对应 questionlist[i]

#%%其他
def remove_sentence_similarity_edges(graph):
    # 找到所有 edge_type 为 'sentence_similarity' 的边
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'sentence_similarity']

    # 删除这些边
    graph.remove_edges_from(edges_to_remove)
    return graph

def remove_edges(graph):
    graph.clear_edges()
    return graph
