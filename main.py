#%%
from utils import common_use
from core.event_segments import event_segment_main
from core.graphconstraction import build_complete_graph
from core.retrival_TOG import TOG_main
from core.answer_rate import get_answer_and_rate
#事件切分
conversation_id=6
chunks=event_segment_main(conversation_id)
graph=build_complete_graph(chunks)

retrival_results=TOG_main(chunks,graph,conversation_id)

#回答与评分
llmanswerlist,rate_all=get_answer_and_rate(retrival_results,conversation_id)

#%%topK检索测试
from core.retrival_others import topk_retival
retrival_results_topK=topk_retival(chunks,6,0)

#%%消融实验-切分

from core import event_segments

conversation_id=2
#turn单元
turn_chunks=event_segments.get_turn_chunks_withtime(conversation_id)
fixsize_chunks_200=event_segments.get_fixsize_chunks_withtime(conversation_id,300)
simple_prompt_chunks=event_segments.event_segment_simple(conversation_id)

def ablation_chunks(chunks):
    graph=build_complete_graph(chunks)
    retrival_results=TOG_main(chunks,graph,conversation_id)
    llmanswerlist,rate_all=get_answer_and_rate(retrival_results,conversation_id)
    return graph,retrival_results,llmanswerlist,rate_all

#消融实验-巩固

import networkx as nx

def graph_without_edge_type(g: nx.Graph, edge_type: str):
    """
    返回一个删除指定类型边后的新图（不修改原图）
    边类型存储在 edge attribute 'type' 中
    """
    g_new = g.copy()

    edges_to_remove = [
        (u, v) for u, v, attrs in g_new.edges(data=True)
        if attrs.get( 'edge_type') == edge_type
    ]

    g_new.remove_edges_from(edges_to_remove)
    return g_new


def ablation_chunks_consolidate(chunks,graph):
    retrival_results=TOG_main(chunks,graph,conversation_id)
    llmanswerlist,rate_all=get_answer_and_rate(retrival_results,conversation_id)
    return graph,retrival_results,llmanswerlist,rate_all



#去除掉整体语义边

graph_wotext=graph_without_edge_type(graph,'text_similarity')
graph_wotext,retrival_results_wotext,llmanswerlist_wotext,rate_all_wotext=ablation_chunks_consolidate(chunks,graph_wotext)
#去除掉元素边
graph_wosentence=graph_without_edge_type(graph,'sentence_similarity')
graph_wosentence,retrival_results_wosentence,llmanswerlist_wosentence,rate_all_wosentence=ablation_chunks_consolidate(chunks,graph_wosentence)


#消融实验-检索
from core.retrival_others import topk_retival,ppr_retival
#topk检索
chunks_pure=[f"time:{i['timestamp']} content:{i['chunk']}" for i in chunks]
chunk_top5=topk_retival(chunks_pure,5,conversation_id)

llmanswerlist_top5,rate_all_top5=get_answer_and_rate(chunk_top5,conversation_id)

#去掉tog
retrival_results_wotog=[i[0:3] for i in retrival_results]
llmanswerlist_wotog,rate_all_wotog=get_answer_and_rate(retrival_results_wotog,conversation_id)
#pagerank

retrival_results_ppr=ppr_retival(chunks_pure,graph,4,6)
llmanswerlist_ppr,rate_all_ppr=get_answer_and_rate(retrival_results_ppr,conversation_id)


#%%主实验其他
conversation_id=6
def pure_chunks(chunks):
    chunks_pure=[f"time:{i['timestamp']} content:{i['chunk']}" for i in chunks]
    return chunks_pure

turn_chunks=pure_chunks(event_segments.get_turn_chunks_withtime(conversation_id))
session_chunks=pure_chunks(event_segments.get_session_chunks_withtime(conversation_id))
fixsize_chunks_100=pure_chunks(event_segments.get_fixsize_chunks_withtime(conversation_id,100))
fixsize_chunks_200=pure_chunks(event_segments.get_fixsize_chunks_withtime(conversation_id,200))
fixsize_chunks_300=pure_chunks(event_segments.get_fixsize_chunks_withtime(conversation_id,300))


def topk_retrival_and_rate(chunks,conversation_id,topK):
    chunk_topk=topk_retival(chunks,topK,conversation_id)
    llmanswerlist_topk,rate_all_topk=get_answer_and_rate(chunk_topk,conversation_id)
    return llmanswerlist_topk,rate_all_topk