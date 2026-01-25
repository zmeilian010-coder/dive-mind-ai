# tools.py
import streamlit as st
import jieba
import re
import datetime
import calendar
from typing import List, Optional, Dict, Any, Tuple
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# 导入自定义模块
import state_manager
import utils
import models
import ai_engine

def get_db():
    """便捷获取数据库的辅助函数"""
    _, _, rag_db = models.get_models()
    return rag_db


@tool
def retrieve_general_knowledge(query: str, keywords: List[str] = None, **kwargs) -> str:
    """带硬核关键词检查的混合检索工具。要求：文档必须至少命中一个关键词才会被采纳。"""
    if not keywords:
        keywords = []
        for key, val in kwargs.items():
            if val and val != "":
                if "departureMonth" in key:
                    keywords.append(f"{val}月")
                    continue
                if isinstance(val, list):
                    keywords.extend([str(item).strip() for item in val if str(item).strip()])
                else:
                    keywords.append(str(val).strip())

    if not keywords:
        keywords = [w for w in jieba.lcut(query) if len(w) > 1]

    print(f"\n[执行高级检索] 提问: {query} | 核心词: {keywords}")

    rag_db = get_db()

    K_RECALL = 30
    # A路：原始语义
    vector_results = rag_db.similarity_search_with_score(query, k=K_RECALL)
    # B路：关键词语义
    kw_query = " ".join(keywords) if keywords else query
    vector_results_kw = rag_db.similarity_search_with_score(kw_query, k=K_RECALL)
    # C路：BM25硬匹配
    all_data = rag_db.get()
    all_docs = [Document(page_content=t, metadata=m or {})
                for t, m in zip(all_data['documents'], all_data['metadatas'])]
    bm25 = BM25Retriever.from_documents(all_docs, preprocess_func=utils.chinese_tokenizer)
    bm25.k = K_RECALL
    tokenized_query = " ".join(utils.chinese_tokenizer(kw_query))
    keyword_docs = bm25.invoke(tokenized_query)

    unique_candidates = {}
    for doc, score in vector_results:
        unique_candidates[doc.page_content[:100]] = {"doc": doc, "v_score": score}
    for doc, score in vector_results_kw:
        cid = doc.page_content[:100]
        if cid not in unique_candidates or score < unique_candidates[cid]["v_score"]:
            unique_candidates[cid] = {"doc": doc, "v_score": score}
    for doc in keyword_docs:
        cid = doc.page_content[:100]
        if cid not in unique_candidates:
            unique_candidates[cid] = {"doc": doc, "v_score": 1.5}

    scored_list = []
    N = len(keywords) if keywords else 0
    KW_UNIT_SCORE = 100 / N if N > 0 else 0

    for content_id, item in unique_candidates.items():
        doc = item["doc"]
        content_lower = doc.page_content.lower()
        matched_words = [kw for kw in keywords if kw.lower() in content_lower] if N > 0 else []

        if N > 0 and not matched_words: continue  # 一票否决

        kw_score = len(matched_words) * KW_UNIT_SCORE
        v_score = max(0, (1.5 - item["v_score"]) / 1.5 * 30)
        total_score = kw_score + v_score

        scored_list.append({"doc": doc, "total_score": total_score, "kw_detail": f"{len(matched_words)}/{N}"})

    scored_list.sort(key=lambda x: x["total_score"], reverse=True)
    winners = scored_list[:10]

    state_manager.DataStorage.BASKET.clear()
    state_manager.DataStorage.BASKET.extend([item["doc"] for item in winners])

    if not winners: return f"Buddy，没捞到包含关键词 '{keywords}' 的资料。"
    return "\n\n".join(
        [f"参考[{i + 1}] (得分:{item['total_score']:.1f}): {item['doc'].page_content[:200]}" for i, item in
         enumerate(winners)])


@tool
def retrieve_boats(query: str, **kwargs) -> str:
    """检索潜水船只 (Boat) 的相关信息。"""
    all_params = {k: v for k, v in kwargs.items() if v is not None}
    all_params["category"] = ["船宿船舶信息"]
    chroma_filters, post_process_filters = utils._build_filter_dict(**all_params)
    rag_db = get_db()
    retriever = rag_db.as_retriever(search_kwargs={"k": 10, "filter": chroma_filters})
    initial_docs = retriever.invoke(query)

    final_docs = []
    if "locationName" in post_process_filters:
        search_term = post_process_filters["locationName"].lower()
        final_docs = [d for d in initial_docs if search_term in str(d.metadata.get("locationName", "")).lower()]
    else:
        final_docs = initial_docs

    final_docs = final_docs[:5]
    state_manager.DataStorage.BASKET.extend(final_docs)
    return utils._format_docs(final_docs)


@tool
def retrieve_trips(query: str, keywords: List[str] = None, **kwargs) -> str:
    """检索具体的船宿行程信息。先硬筛选，再执行关键词评分。"""
    print(f"\n[retrieve_trips船宿检索] Query: '{query}' | 关键词: {keywords}| 其他参数：{kwargs}")

    final_search_keywords = keywords if keywords else [w for w in jieba.lcut(query) if len(w) > 1]
    loc = kwargs.get("locationName")
    if loc and loc not in final_search_keywords: final_search_keywords.append(loc)
    print(f"📍 最终用于内容检索的 Keywords: {final_search_keywords}")

    # 1. 构建 Filter
    filter_list = [{"category": {"$eq": "船宿行程"}}]
    if "allowed_levels" in kwargs:
        filter_list.append({"certification": {"$in": kwargs["allowed_levels"]}})

    raw_exp = kwargs.get("max_experience", 0)
    try:
        final_exp_val = float(str(raw_exp).split("-")[-1]) if "-" in str(raw_exp) else float(
            re.sub(r'[^\d.]', '', str(raw_exp)) or 0)
    except:
        final_exp_val = 0

    if final_exp_val > 0:
        filter_list.append({"experience": {"$lte": int(final_exp_val)}})
    print(f"✅ 经验过滤生效：查找要求经验 <= {int(final_exp_val)} 瓶的行程")

    if kwargs.get("needs_nitrox"): filter_list.append({"nitrox": {"$ne": "No"}})
    if kwargs.get("needs_wifi"): filter_list.append({"wifi": {"$ne": "No"}})

    if kwargs.get("departureMonth"):
        y, m = utils.get_correct_year_and_month(kwargs["departureMonth"], kwargs.get("departureYear"))
        start_ts, end_ts = utils.calculate_buffered_range(y, m)
        filter_list.append({"departureDate": {"$gte": start_ts}, "departureDate": {"$lte": end_ts}})

    chroma_filters = {"$and": filter_list}
    print(f"📍 环节 2 (Metadata 硬过滤): {chroma_filters}")

    # 2. 检索
    rag_db = get_db()

    all_data = rag_db.get(where=chroma_filters)
    if not all_data or not all_data.get('documents'): return "Buddy，没找到匹配行程。"

    candidate_docs = [Document(page_content=t, metadata=m) for t, m in
                      zip(all_data['documents'], all_data['metadatas'])]
    bm25 = BM25Retriever.from_documents(candidate_docs, preprocess_func=lambda x: jieba.lcut(x))
    keyword_docs = bm25.invoke(" ".join(final_search_keywords))

    # 3. 计分
    scored_list = []
    UNIT_SCORE = 100 / (len(final_search_keywords) or 1)

    print("\n--- 船宿精排计分分析 ---")
    for doc in keyword_docs:
        content_low = doc.page_content.lower()
        meta = doc.metadata
        total_score = 0
        for kw in final_search_keywords:
            kw_l = kw.lower()
            if kw_l in str(meta.get('tour_nameCN', '')).lower() or kw_l in str(meta.get('locationName', '')).lower():
                total_score += UNIT_SCORE
            elif kw_l in content_low:
                total_score += (UNIT_SCORE * 0.6)
        if total_score > 0: scored_list.append({"doc": doc, "score": total_score})

    scored_list.sort(key=lambda x: x["score"], reverse=True)
    winners = []
    seen_boats = set()
    for item in scored_list:
        bid = item["doc"].metadata.get("boatId")
        if bid not in seen_boats:
            winners.append(item)
            seen_boats.add(bid)
        if len(winners) >= 5: break



    state_manager.DataStorage.BASKET.extend([w["doc"] for w in winners])
    print(f">>> 最终选出 {len(winners)} 艘不同的船只资料")
    if not winners:
        return "Buddy，没找到匹配的行程。"

    # 4. 回传
    formatted_output = []
    final_docs_for_basket = []
    for i, winner in enumerate(winners):
        doc = winner["doc"]
        parsed_details = utils.parse_trip_content(doc.page_content)
        meta = doc.metadata

        # 合并字段
        doc.metadata.update(parsed_details)
        final_docs_for_basket.append(doc)

        # 构造给 AI 的摘要
        ai_summary = f"""
        船宿方案 [{i + 1}]:
        - 船名: {meta.get('boat_nameEN')}(parsed_details['boat_nameCN'])
        - 目的地: {meta.get('locationName', '未知')}
        - 准入要求: 证书需 {meta.get('certification', 'OW')} | 经验需 {meta.get('experience', 0)} 瓶以上
        - 价格预估: {parsed_details['price_str']} | 空位（非实时）: {parsed_details['available_count']}
        - 航期: {meta.get('departureDate_display', '未知')} 至 {meta.get('arrivalDate_display', '未知')}
        - 潜水约 {parsed_details['dives']} 次
        - 船上设施: 高氧({parsed_details['nitrox']}) | Wi-Fi({parsed_details['wifi']}) | 技术潜水({parsed_details['tech_friendly']})
        - 预订政策: {parsed_details['booking_policy']}
        - 取消政策: {parsed_details['cancellation_policy']}
        - 点评: 此方案匹配度 {winner['score']:.1f}，命中了关键词 {keywords}
        """
        formatted_output.append(ai_summary)

    # 存入保险箱
    state_manager.DataStorage.BASKET.extend(final_docs_for_basket)

    return "\n\n".join(formatted_output)


@tool
def retrieve_tours(query: str, **kwargs) -> str:
    """检索潜水路线 (Tour) 的相关信息。"""
    print(f"\n[Agent正在调用 retrieve_tours 工具，查询: '{query}']")

    all_params = {k: v for k, v in kwargs.items() if v is not None and k not in ['query']}
    all_params["category"] = ["船宿路线"]
    chroma_filters, post_process_filters = utils._build_filter_dict(**all_params)
    print(f"[工具内部已强制锁定参数 category: ['船宿路线']]")
    # 2. 检索
    rag_db = get_db()
    retriever = rag_db.as_retriever(search_kwargs={"k": 10, "filter": chroma_filters})
    initial_docs = retriever.invoke(query)

    final_docs = initial_docs
    if "locationName" in post_process_filters:
        st_lower = post_process_filters["locationName"].lower()
        final_docs = [d for d in initial_docs if st_lower in str(d.metadata.get("locationName", "")).lower()]

    final_docs = final_docs[:5]
    state_manager.DataStorage.BASKET.extend(final_docs)
    return utils._format_docs(final_docs)