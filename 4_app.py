__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from typing import List


# --- 1. 这里粘贴你脚本 B 顶部的所有 import 语句 ---
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
from pathlib import Path
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables.base import Runnable
from langchain.tools import tool

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import datetime
import calendar
from typing import Optional, List, Dict, Any, Tuple

load_dotenv()


# --- 2. 这里粘贴你脚本 B 里定义的所有工具 (@tool) 和变量 ---
# =======================================================
# RAG 系统配置
# =======================================================
CHROMA_PATH = "chroma"
# LOCAL_BGE_M3_MODEL_PATH = Path("E:/Python项目/dify应用的评估效果/local_bge_m3_model/bge-m3")  云部署版本不用本地嵌入模型，改用硅基流动的云服务
# RAG_EMBEDDING_MODEL_NAME = str(LOCAL_BGE_M3_MODEL_PATH)

RAG_LLM_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


# =======================================================
# 辅助函数
# =======================================================
def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# =======================================================
# 元数据列名映射 (与 ChromaDB 中实际存储的元数据键名一致)
# 这里确保工具参数名和 ChromaDB 实际键名一致，简化映射
# =======================================================
METADATA_CHROMA_KEYS = {
    "Metadata_source": "Metadata_source",
    "Metadata_file_type": "Metadata_file_type",
    "Metadata_row_number": "Metadata_row_number",
    "Metadata_Header1": "Metadata_Header1",
    "category": "category",
    "boatId": "boatId",
    "tourId": "tourId",
    "tripId": "tripId",
    "nameCN": "nameCN",
    "nameEN": "nameEN",
    "locationName": "locationName",
    "arrivalDate": "arrivalDate",
    "departureDate": "departureDate",
    "updatedTime": "updatedTime",
    "experience": "experience",
    "certification": "certification",
    "dives": "dives",
    "duration": "duration",
    "nights": "nights",
    "nitrox": "nitrox",
    "wifi": "wifi",
    "diving_equipment": "diving_equipment",
    "tech_diving_friendly": "tech_diving_friendly",
    "languages": "languages",
    "policy": "policy",
    "rating": "rating",
    "yearBuilt": "yearBuilt"
}

# =======================================================
# 初始化 RAG 组件 (LLM 和 ChromaDB)
# =======================================================
if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。请在 .env 文件中配置。")

agent_llm = ChatOpenAI(
    model=RAG_LLM_MODEL,
    openai_api_base=DEEPSEEK_BASE_URL,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.0
)


embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",  # 必须是这个全名
    api_key=os.getenv("SILICONFLOW_API_KEY"),  # 注意：在新版里是 api_key，不是 openai_api_key
    base_url="https://api.siliconflow.cn/v1"    # 注意：在新版里是 base_url，不是 openai_api_base
)

if not os.path.exists(CHROMA_PATH):
    raise ValueError(f"ChromaDB 路径 '{CHROMA_PATH}' 不存在。请确保已运行 ingest.py 创建了知识库。")
rag_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


# =======================================================
# 定义 Agent 的专用检索工具
# =======================================================

def _build_filter_dict(**params) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print(f"\n_build_filter_dict 接收到的参数: {params}")

    filter_list = []
    post_process_filters = {}
    current_year = datetime.datetime.now().year

    for key, value in params.items():
        if value is None: continue

        # 1. 后处理字段 (字符串匹配)
        if key == "locationName":
            post_process_filters["locationName"] = str(value)
            continue

        # 2. Category 列表匹配
        if key == "category" and isinstance(value, list):
            if value:
                filter_list.append({"category": {"$in": value}})
            continue

        # 3. 处理月份简写 (departureMonth -> 时间戳范围)
        if key == "departureMonth":
            try:
                month = int(value)
                if 1 <= month <= 12:
                    start_dt = datetime.datetime(current_year, month, 1, 0, 0, 0)
                    last_day = calendar.monthrange(current_year, month)[1]
                    end_dt = datetime.datetime(current_year, month, last_day, 23, 59, 59)

                    # 这里的 .timestamp() 对应数据库里的 float 类型
                    filter_list.append({"departureDate": {"$gte": start_dt.timestamp()}})
                    filter_list.append({"departureDate": {"$lte": end_dt.timestamp()}})
                    print(f"  -> departureMonth 转换为时间戳范围")
            except Exception as e:
                print(f"警告: departureMonth 处理出错: {e}")
            continue

        # 4. 处理带后缀的操作符 (_gt, _gte, _lt, _lte 等)
        op_mapping = {
            '_eq': '$eq', '_ne': '$ne', '_gt': '$gt', '_gte': '$gte',
            '_lt': '$lt', '_lte': '$lte', '_in': '$in', '_nin': '$nin'
        }
        found_op = False

        for op_suffix, chroma_op in op_mapping.items():
            if key.endswith(op_suffix):
                field_name = key[:-len(op_suffix)]
                processed_value = value

                # === 【核心修改区域】针对日期字段的特殊处理 ===
                if field_name in ['departureDate', 'arrivalDate']:
                    try:
                        # AI 可能会传 "2026-01-01" 或 "2026-01-01T12:00:00Z"
                        str_val = str(value).replace("Z", "")  # 去掉可能导致解析失败的 Z

                        # 解析为日期对象
                        dt = datetime.datetime.fromisoformat(str_val)

                        # 转换为时间戳数字 (Float)，这就跟数据库里的格式对上了！
                        processed_value = dt.timestamp()
                        print(f"  -> 将 {key}='{value}' 转换为时间戳: {processed_value}")
                    except Exception as e:
                        print(f"警告: 日期参数 {key}='{value}' 转换时间戳失败: {e}")
                        # 如果转换失败，保留原值，虽然可能查不到，但至少不崩
                        pass

                # 处理其他数字字段
                elif field_name in ['duration', 'dives', 'nights', 'rating']:
                    try:
                        processed_value = float(value)
                    except:
                        pass

                filter_list.append({field_name: {chroma_op: processed_value}})
                found_op = True
                break

        if found_op: continue

        # 5. 处理剩下的精确匹配 (Exact Match)
        if key not in post_process_filters and key != 'departureMonth':
            final_val = value
            # 如果 AI 居然传了个精确日期 (比如 departureDate="2026-01-01")
            # 我们也要把它转成时间戳，否则肯定查不到
            if key in ['departureDate', 'arrivalDate']:
                try:
                    str_val = str(value).replace("Z", "")
                    dt = datetime.datetime.fromisoformat(str_val)
                    final_val = dt.timestamp()
                except:
                    pass

            filter_list.append({key: final_val})

    # 打包返回
    final_where = {}
    if len(filter_list) == 1:
        final_where = filter_list[0]
    elif len(filter_list) > 1:
        final_where = {"$and": filter_list}

    print(f"最终修正后的发送条件 (数字版): {final_where}")
    return final_where, post_process_filters

# 辅助函数：格式化文档输出给LLM (保持不变)
def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "未找到相关信息。"
    formatted_list = []
    for i, doc in enumerate(docs):
        content = doc.page_content.replace('\n', ' ').strip()
        metadata_display = {k: v for k, v in doc.metadata.items()
                            if k not in ['source', 'timestamp', 'file_type', 'project', 'processed_by', 'original_source']}
        metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata_display.items()])
        formatted_list.append(f"文档 {i+1}:\n内容: {content}\n元数据: {metadata_str}\n---")
    return "\n".join(formatted_list)


@tool
def retrieve_tours(
        query: str,
        tourId: Optional[str] = None,
        tripId: Optional[str] = None,
        locationName: Optional[str] = None,
        nameCN: Optional[str] = None,
        nameEN: Optional[str] = None,
        boatId: Optional[str] = None,
        experience_gt: Optional[int] = None, # 路线的潜水经验要求大于此值。
        experience_gte: Optional[int] = None, # 路线的潜水经验要求大于或等于此值。
        experience_lt: Optional[int] = None, # 路线的潜水经验要求小于此值。
        experience_lte: Optional[int] = None, # 路线的潜水经验要求小于或等于此值。
        certification: Optional[str] = None,
        dives_gt: Optional[int] = None,  # 新增：潜水次数大于
        dives_gte: Optional[int] = None,  # 新增：潜水次数大于等于
        dives_lt: Optional[int] = None,
        dives_lte: Optional[int] = None,
        duration_gt: Optional[int] = None,  # 新增：时长大于
        duration_gte: Optional[int] = None,
        duration_lt: Optional[int] = None,
        duration_lte: Optional[int] = None,
        nights_gt: Optional[int] = None,  # 新增：夜晚时长大于
        nights_gte: Optional[int] = None,
        nights_lt: Optional[int] = None,
        nights_lte: Optional[int] = None,
        departureMonth: Optional[int] = None,  # 新增：出发月份
        **kwargs  # 捕获其他未知参数
) -> str:
    """
    检索潜水路线 (Tour) 的相关信息。
    当用户询问关于特定路线ID、行程ID、地点、或路线名称时调用此工具。
    可选参数：
    - tourId (str): 潜水路线的唯一ID。
    - tripId (str): 潜水行程的唯一ID (路线可能包含行程)。
    - locationName (str): 路线所在地点。此参数将进行“包含”匹配（不区分大小写）。
    - nameCN (str): 路线的中文名称。
    - nameEN (str): 路线的英文名称。
    - boatId (str): 执行路线的船舶的唯一ID。
    - experience_gt (int): 路线的潜水经验要求大于此值。
    - experience_gte (int): 路线的潜水经验要求大于或等于此值。
    - experience_lt (int): 路线的潜水经验要求小于此值。
    - experience_lte (int): 路线的潜水经验要求小于或等于此值。
    - certification(str): 路线的潜水证书要求。
    - dives_gt (int): 路线潜水次数大于此值。
    - dives_gte (int): 路线潜水次数大于或等于此值。
    - dives_lt (int): 路线潜水次数小于此值。
    - dives_lte (int): 路线潜水次数小于或等于此值。
    - duration_gt (int): 路线时长（天数）大于此值。
    - duration_gte (int): 路线时长（天数）大于或等于此值。
    - duration_lt (int): 路线时长（天数）小于此值。
    - duration_lte (int): 路线时长（天数）小于或等于此值。
    - nights_gt (int): 路线夜晚时长大于此值。
    - nights_gte (int): 路线夜晚时长大于或等于此值。
    - nights_lt (int): 路线夜晚时长小于此值。
    - nights_lte (int): 路线夜晚时长小于或等于此值。
    - departureMonth (int): 路线出发的月份（1-12）。工具将自动转换为当前年份的日期范围。
    """
    print(f"\n[Agent正在调用 retrieve_tours 工具，查询: '{query}']")

    # 1. 提取所有参数
    all_params_for_filter = {k: v for k, v in locals().items()
                             if k not in ['query', 'kwargs', 'self'] and v is not None}
    all_params_for_filter.update(kwargs)

    # === 【核心修复点 1】在构建之前，强制锁定 category ===
    # 这样生成的字典会自动包含在 $and 逻辑中，不会产生顶级操作符冲突
    all_params_for_filter["category"] = ["船宿路线"]
    print(f"[工具内部已强制锁定参数 category: ['船宿路线']]")

    # 2. 调用构建函数（使用我们之前改好的、能自动包 $and 的版本）
    chroma_filters, post_process_filters = _build_filter_dict(**all_params_for_filter)

    print(f"[检索工具将尝试使用以下合法的 ChromaDB 过滤条件: {chroma_filters}]")

    # 3. 使用生成的 chroma_filters 进行检索
    # 注意：不要再在这里手动修改 chroma_filters 了
    retriever = rag_db.as_retriever(search_kwargs={"k": 10, "filter": chroma_filters})
    initial_docs: List[Document] = retriever.invoke(query)

    # 4. 后处理逻辑 (locationName 包含匹配)
    final_docs: List[Document] = []
    if "locationName" in post_process_filters:
        search_term_lower = post_process_filters["locationName"].lower()
        for doc in initial_docs:
            doc_location_name = doc.metadata.get("locationName")
            if doc_location_name and search_term_lower in str(doc_location_name).lower():
                final_docs.append(doc)
        print(f"[对检索结果进行 locationName 包含 '{post_process_filters['locationName']}' 的后处理筛选]")
    else:
        final_docs = initial_docs

    if not final_docs:
        print("[retrieve_tours 工具未找到相关路线信息。]")
        return "未找到相关路线信息。"

    final_docs = final_docs[:5]

    # === 【新增：把格式化前的原始文档传给前端渲染卡片】 ===
    import streamlit as st
    # 检查 session_state 里有没有这个“篮子”，没有就建一个
    if "last_retrieved_docs" not in st.session_state:
        st.session_state.last_retrieved_docs = []

    # 把这次找到的 final_docs 整个放进“篮子”里
    # 注意：我们存的是原始的 Document 对象列表，包含了所有的元数据
    st.session_state.last_retrieved_docs.extend(final_docs)
    # =====================================

    return _format_docs(final_docs)


@tool
def retrieve_trips(
        query: str,
        tripId: Optional[str] = None,  # tripId 常常是精确匹配
        tourId: Optional[str] = None,
        boatId: Optional[str] = None,
        locationName: Optional[str] = None,  # 包含匹配，需要后处理
        nameCN: Optional[str] = None,
        nameEN: Optional[str] = None,
        category: Optional[List[str]] = None,  # 增加 category 过滤
        arrivalDate_gte: Optional[str] = None,  # 抵达日期 >= (ISO格式日期字符串)
        arrivalDate_lte: Optional[str] = None,  # 抵达日期 <= (ISO格式日期字符串)
        departureDate_gte: Optional[str] = None,  # 出发日期 >= (ISO格式日期字符串)
        departureDate_lte: Optional[str] = None,  # 出发日期 <= (ISO格式日期字符串)
        arrivalMonth: Optional[int] = None,  # 抵达月份
        departureMonth: Optional[int] = None,  # 出发月份
        updatedTime_gte: Optional[str] = None,  # 更新时间 >= (ISO格式日期字符串)
        updatedTime_lte: Optional[str] = None,  # 更新时间 <= (ISO格式日期字符串)
        duration_gt: Optional[int] = None,  # 时长大于
        duration_gte: Optional[int] = None,  # 时长大于等于
        duration_lt: Optional[int] = None,  # 时长小于
        duration_lte: Optional[int] = None,
        nights_gt: Optional[int] = None,  # 夜晚时长大于
        nights_gte: Optional[int] = None,
        nights_lt: Optional[int] = None,
        nights_lte: Optional[int] = None,
        **kwargs  # 捕获其他未知参数
) -> str:
    """
    检索潜水行程 (Trip) 的相关信息。
    当用户询问关于特定行程ID、路线ID、船只ID、地点、或行程名称时调用此工具。
    行程是路线的具体一次出游。

    可选参数：
    - tripId (str): 潜水行程的唯一ID。
    - tourId (str): 行程的潜水路线ID。
    - boatId (str): 行程所使用的船只ID。
    - locationName (str): 行程所在的地点。此参数将进行“包含”匹配（不区分大小写）。
    - nameCN (str): 行程的中文名称。
    - nameEN (str): 行程的英文名称。
    - category (List[str]): 文档的内容类型列表，例如 ["船宿行程"]。
    - arrivalDate_gte (str): 返程抵达日期大于或等于此ISO格式日期字符串。
    - arrivalDate_lte (str): 返程抵达日期小于或等于此ISO格式日期字符串。
    - departureDate_gte (str): 启程出发日期大于或等于此ISO格式日期字符串。
    - departureDate_lte (str): 启程出发日期小于或等于此ISO格式日期字符串。
    - arrivalMonth (int): 行程抵达的月份（1-12）。工具将自动转换为当前年份的日期范围。
    - departureMonth (int): 行程出发的月份（1-12）。工具将自动转换为当前年份的日期范围。
    - updatedTime_gte (str): 船宿信息更新时间大于或等于此ISO格式日期字符串。
    - updatedTime_lte (str): 船宿信息更新时间小于或等于此ISO格式日期字符串。
    - duration_gt (int): 行程时长（天数）大于此值。
    - duration_gte (int): 行程时长（天数）大于或等于此值。
    - duration_lt (int): 行程时长（天数）小于此值。
    - duration_lte (int): 行程时长（天数）小于或等于此值。
    - nights_gt (int): 行程夜晚时长大于此值。
    - nights_gte (int): 行程夜晚时长大于或等于此值。
    - nights_lt (int): 行程夜晚时长小于此值。
    - nights_lte (int): 行程夜晚时长小于或等于此值。
    """
    print(f"\n[Agent正在调用 retrieve_trips 工具，查询: '{query}']")

    # 1. 提取所有参数
    all_params_for_filter = {k: v for k, v in locals().items()
                             if k not in ['query', 'kwargs', 'self'] and v is not None}
    all_params_for_filter.update(kwargs)

    # === 【核心修复点 1】在构建之前，强制锁定 category ===
    # 这样生成的字典会自动包含在 $and 逻辑中，不会产生顶级操作符冲突
    all_params_for_filter["category"] = ["船宿行程"]
    print(f"[工具内部已强制锁定参数 category: ['船宿行程']]")

    # 2. 调用构建函数（使用我们之前改好的、能自动包 $and 的版本）
    chroma_filters, post_process_filters = _build_filter_dict(**all_params_for_filter)

    print(f"[检索工具将尝试使用以下合法的 ChromaDB 过滤条件: {chroma_filters}]")
    # ====================================================================

    retriever = rag_db.as_retriever(search_kwargs={"k": 10, "filter": chroma_filters})  # 调大k以应对后处理
    initial_docs: List[Document] = retriever.invoke(query)

    final_docs: List[Document] = []
    if "locationName" in post_process_filters:
        search_term_lower = post_process_filters["locationName"].lower()
        for doc in initial_docs:
            doc_location_name = doc.metadata.get("locationName")
            if doc_location_name and search_term_lower in str(doc_location_name).lower():
                final_docs.append(doc)
        print(
            f"[对检索结果进行 locationName 包含 '{post_process_filters['locationName']}' 的后处理筛选 (不区分大小写)]")
    else:
        final_docs = initial_docs

    if not final_docs:
        print("[retrieve_trips 工具未找到相关行程信息。]")
        return "未从知识库中检索到相关行程信息。"

    # 限制最终返回给 Agent 的文档数量，例如回到 k=5
    final_docs = final_docs[:5]

    # === 【新增：把格式化前的原始文档传给前端渲染卡片】 ===
    import streamlit as st
    # 检查 session_state 里有没有这个“篮子”，没有就建一个
    if "last_retrieved_docs" not in st.session_state:
        st.session_state.last_retrieved_docs = []

    # 把这次找到的 final_docs 整个放进“篮子”里
    # 注意：我们存的是原始的 Document 对象列表，包含了所有的元数据
    st.session_state.last_retrieved_docs.extend(final_docs)
    # =====================================

    return _format_docs(final_docs)


@tool
def retrieve_boats(
        query: str,
        boatId: Optional[str] = None,
        nameCN: Optional[str] = None,
        nameEN: Optional[str] = None,
        category: Optional[List[str]] = None,  # 增加 category 过滤
        rating_gt: Optional[float] = None,  # 评分大于
        rating_gte: Optional[float] = None,
        rating_lt: Optional[float] = None,
        rating_lte: Optional[float] = None,
        yearBuilt_gt: Optional[int] = None,  # 建造年份大于
        yearBuilt_gte: Optional[int] = None,
        yearBuilt_lt: Optional[int] = None,
        yearBuilt_lte: Optional[int] = None,
        nitrox: Optional[str] = None,  # bool值可能需要Agent判断转为 "True" 或 "False" 字符串
        wifi: Optional[str] = None,
        diving_equipment: Optional[str] = None,
        tech_diving_friendly: Optional[str] = None,
        languages: Optional[str] = None,
        policy: Optional[str] = None,
        **kwargs  # 捕获其他未知参数
) -> str:
    """
    检索潜水船只 (Boat) 的相关信息。
    当用户询问关于特定船只ID、船只名称、评分、建造年份、设施等信息时调用此工具。

    可选参数：
    - boatId (str): 船只的唯一ID。
    - nameCN (str): 船只的中文名称。
    - nameEN (str): 船只的英文名称。
    - category (List[str]): 文档的内容类型列表，例如 ["船宿船舶信息"]。
    - rating_gt (float): 船只评分大于此值。
    - rating_gte (float): 船只评分大于或等于此值。
    - rating_lt (float): 船只评分小于此值。
    - rating_lte (float): 船只评分小于或等于此值。
    - yearBuilt_gt (int): 船只建造年份大于此值。
    - yearBuilt_gte (int): 船只建造年份大于或等于此值。
    - yearBuilt_lt (int): 船只建造年份小于此值。
    - yearBuilt_lte (int): 船只建造年份小于或等于此值。
    - nitrox (str): 船宿是否支持高氧。
    - wifi (str): 船宿是否支持WiFi 。
    - diving_equipment (str): 船宿是否提供装备。
    - tech_diving_friendly (str): 船宿是否支持技术潜水。
    - languages (str): 船宿支持的语言种类。
    - policy (str): 船宿的预订、退款政策。
    """
    print(f"\n[Agent正在调用 retrieve_boats 工具，查询: '{query}']")

    # 1. 提取所有参数
    all_params_for_filter = {k: v for k, v in locals().items()
                             if k not in ['query', 'kwargs', 'self'] and v is not None}
    all_params_for_filter.update(kwargs)

    # === 【核心修复点 1】在构建之前，强制锁定 category ===
    # 这样生成的字典会自动包含在 $and 逻辑中，不会产生顶级操作符冲突
    all_params_for_filter["category"] = ["船宿船舶信息"]
    print(f"[工具内部已强制锁定参数 category: ['船宿船舶信息']]")

    # 2. 调用构建函数（使用我们之前改好的、能自动包 $and 的版本）
    chroma_filters, post_process_filters = _build_filter_dict(**all_params_for_filter)

    print(f"[检索工具将尝试使用以下合法的 ChromaDB 过滤条件: {chroma_filters}]")
    # ====================================================================

    retriever = rag_db.as_retriever(search_kwargs={"k": 10, "filter": chroma_filters})  # 调大k以应对后处理
    initial_docs: List[Document] = retriever.invoke(query)

    final_docs: List[Document] = []
    # 船只信息可能也需要 locationName 后处理，如果你的船只元数据里有 locationName 字段
    # 如果 boat 没有 locationName 元数据，这部分可以省略或改为其他后处理逻辑
    if "locationName" in post_process_filters:
        search_term_lower = post_process_filters["locationName"].lower()
        for doc in initial_docs:
            doc_location_name = doc.metadata.get("locationName")  # 假设船只有 locationName
            if doc_location_name and search_term_lower in str(doc_location_name).lower():
                final_docs.append(doc)
        print(
            f"[对检索结果进行 locationName 包含 '{post_process_filters['locationName']}' 的后处理筛选 (不区分大小写)]")
    else:
        final_docs = initial_docs

    if not final_docs:
        print("[retrieve_boats 工具未找到相关船只信息。]")
        return "未从知识库中检索到相关船只信息。"

    # 限制最终返回给 Agent 的文档数量，例如回到 k=5
    final_docs = final_docs[:5]

    # === 【新增：把格式化前的原始文档传给前端渲染卡片】 ===
    import streamlit as st
    # 检查 session_state 里有没有这个“篮子”，没有就建一个
    if "last_retrieved_docs" not in st.session_state:
        st.session_state.last_retrieved_docs = []

    # 把这次找到的 final_docs 整个放进“篮子”里
    # 注意：我们存的是原始的 Document 对象列表，包含了所有的元数据
    st.session_state.last_retrieved_docs.extend(final_docs)
    # =====================================

    return _format_docs(final_docs)


# 可以添加一个通用工具用于检索其他类别或无法明确归类的知识
@tool
def retrieve_general_knowledge(
        query: str,
        category: str = None,
        Metadata_source: str = None,
        Metadata_file_type: str = None,
        Metadata_Header1: str = None,
        **kwargs
) -> str:
    """
    检索潜水相关的通用知识，例如潜水装备、安全守则、潜水技巧等。
    当用户问题无法明确归类到路线、行程或船只时，或者其他工具没有检索到内容时，调用此工具。
    可选参数：
    - category (str): 文档的内容类型，例如 '潜水装备', '安全守则'。
    - Metadata_source (str): 信息来源的文件路径（如 'OW教材.md'）。
    - Metadata_file_type (str): 文件类型（如 'md', 'xlsx'）。
    - Metadata_Header1 (str): Markdown 文档的一级标题。
    """
    print(f"\n[Agent正在调用 retrieve_general_knowledge 工具，查询: '{query}']")

    # 收集所有可能的过滤参数
    all_params_for_filter = {k: v for k, v in locals().items()
                             if k not in ['query', 'kwargs', 'self'] and v is not None}
    all_params_for_filter.update(kwargs)  # 合并 kwargs

    chroma_filters, post_process_filters = _build_filter_dict(**all_params_for_filter)

    print(f"[检索工具将尝试使用以下ChromaDB过滤条件: {chroma_filters}]")

    search_kwargs = {"k": 5}
    # === 关键修改 ===
    # 只有当 chroma_filters 非空时，才添加 filter 参数
    if chroma_filters:
        search_kwargs["filter"] = chroma_filters
    # =================

    retriever = rag_db.as_retriever(search_kwargs=search_kwargs)
    initial_docs: List[Document] = retriever.invoke(query)

    # 如果 post_process_filters 中有其他需要后处理的字段，这里需要添加相应逻辑
    final_docs = initial_docs  # 在没有 locationName 后处理的情况下，直接使用 initial_docs

    if not final_docs:
        print("[retrieve_general_knowledge 工具未找到相关通用知识。]")
        return "未从知识库中检索到相关通用知识。"

    # === 【新增：把格式化前的原始文档传给前端渲染卡片】 ===
    import streamlit as st
    # 检查 session_state 里有没有这个“篮子”，没有就建一个
    if "last_retrieved_docs" not in st.session_state:
        st.session_state.last_retrieved_docs = []

    # 把这次找到的 final_docs 整个放进“篮子”里
    # 注意：我们存的是原始的 Document 对象列表，包含了所有的元数据
    st.session_state.last_retrieved_docs.extend(final_docs)
    # =====================================

    formatted_docs = format_docs(final_docs)
    print(f"[retrieve_general_knowledge 工具返回了 {len(final_docs)} 个文档，内容片段: {formatted_docs[:200]}...]")
    return formatted_docs


# --- 3. 封装 Agent 初始化 ---
@st.cache_resource  # 核心：保证 Agent 只在启动时创建一次
def get_agent():
    # 这里粘贴你刚才给我的那段构建逻辑
    # =======================================================
    # 构建 LangGraph Agent
    # =======================================================
    # 代理的提示模板，用于引导Agent的思考和工具使用
    AGENT_PROMPT_STR = """
    你是一个专业的潜水知识问答助手。你的任务是根据用户的提问，通过调用合适的工具从潜水知识库中获取信息，然后提供准确的答案。

    你有以下工具可以使用：
    {tools}

    **核心检索策略 (请务必严格遵循):**

    1.  **判断用户问题类别：**
        *   **如果用户的问题明确与“船宿”相关**（例如，询问船宿路线、行程、船只、船宿价格、船宿地点、预订船宿等），请首先进行船宿相关的知识检索。将 `category` 参数设置为 `["船宿船舶信息", "船宿路线", "船宿行程"]` (注意这是一个列表)，确保检索范围限定在船宿相关文档，并将用户问题的核心词作为 `query`；第二步进行通用潜水知识检索。
        *   **如果用户的问题与“船宿”无关，而是其他通用潜水知识**（例如，潜水装备、潜水技巧、潜水地点、潜水证书、海洋生物等），则进行通用潜水知识检索。

    2.  **参数提取与转换规则 (非常重要，请仔细学习并应用):**
        *   **地点包含匹配：** 当用户询问路线或行程的地点时（如“南极的船宿”），请使用 `locationName` 参数。工具会自动进行包含匹配。
        *   **月份日期筛选：** 当用户提及某个月份的出发日期时（如“11月出发的路线”），请使用 `departureMonth` 参数，其值为对应的月份数字（1-12）。工具会自动转换为该月份的日期范围。
            *   **示例:** “11月” -> `departureMonth=11`
        *   **天数/次数范围筛选：** 当用户询问“X天以上”、“不少于X天”、“不超过X次潜水”等范围条件时，请使用带有 `_gt` (大于), `_gte` (大于等于), `_lt` (小于), `_lte` (小于等于) 后缀的参数。
            *   **示例 1 (天数时长):** “8天以上的行程” -> `duration_gt=8`
            *   **示例 2 (潜水次数):** “不少于5次的潜水路线” -> `dives_gte=5`
            *   **示例 3 (夜晚时长):** “小于3晚的船宿” -> `nights_lt=3`
        *   **默认数量：** 如果用户未指定具体数量，工具默认检索5个最相关的文档。

    3.  **船宿类问题检索（优先）：**
        *   **Thought:** 用户的问题是关于船宿的。我需要调用 `retrieve_tours` 或 `retrieve_trips` 工具，并根据问题中提取的参数进行精确过滤和范围过滤。我**必须**将 `category` 参数设置为 `["船宿船舶信息", "船宿路线", "船宿行程"]` (注意这是一个列表)，并将用户问题的核心词作为 `query`。
            *   **示例 1 (月份筛选):**
                *   用户问：“我想找11月出发的南极船宿路线。”
                *   **Thought:** 用户在问11月出发的南极船宿路线。这是一个船宿类问题。我需要调用 `retrieve_tours` 工具，并设置 `departureMonth` 为11，`locationName` 为“南极”，并过滤 `category`。
                *   **Action:** `retrieve_tours(query="11月出发的南极船宿路线", departureMonth=11, locationName="南极", category=["船宿船舶信息", "船宿路线", "船宿行程"])`
                *   **Observation:** [检索工具找到相关文档...]
                *   **Final Answer:** [根据文档回答...]
            *   **示例 2 (时长筛选):**
                *   用户问：“有没有8天以上的潜水行程？”
                *   **Thought:** 用户在问8天以上的潜水行程。这是一个船宿类问题。我需要调用 `retrieve_trips` 工具，并设置 `duration_gt` 为8，并过滤 `category`。
                *   **Action:** `retrieve_trips(query="8天以上的潜水行程", duration_gt=8, category=["船宿船舶信息", "船宿路线", "船宿行程"])`
                *   **Observation:** [检索工具找到相关文档...]
                *   **Final Answer:** [根据文档回答...]
        *   **多步推理：** 在船宿类问题中，如果初步检索到的文档包含 `boatId` 或 `tourId` 等关联ID，并且用户问题需要进一步的船只或行程信息，**请参照“知识库关联规则”进行多步推理**。

    4.  **通用潜水知识检索（其他情况）：**
        *   **Thought:** 用户的问题不涉及船宿，是通用潜水知识。我可以直接调用 `retrieve_general_knowledge` 工具进行语义检索，不需要额外的 `category` 过滤。
        *   **Action:** 调用 `retrieve_general_knowledge` 工具，只提供 `query` 参数，不设置 `category` 或其他特殊过滤参数。
            *   **示例：**
                *   用户问：“潜水需要什么装备？”
                *   **Thought:** 用户在问潜水装备，这是通用潜水知识。我需要调用 `retrieve_general_knowledge` 工具进行语义检索。
                *   **Action:** `retrieve_general_knowledge(query="潜水需要什么装备")`
                *   **Observation:** [检索工具找到相关文档...]
                *   **Final Answer:** [根据文档回答...]

    **知识库关联规则 (在船宿类问题中需要多步推理时，请务必遵循):**
    - **路线信息 (tourId), 船宿行程信息(tripId) 和船只信息 (boatId) 之间通过 'boatId' 进行关联。**
      如果你通过检索信息（使用 tourId 或 tripId 过滤），在检索结果中发现了一个 **boatId**，而用户的问题需要关于这艘船的更多详情（例如船名、评分、建造年份等），你必须执行以下多步检索：
      1.  首先，使用 `retrieve_tours` 或 `retrieve_trips` 工具，通过用户提供的路线ID（`tourId` 或 `tripId`）和相关查询来获取路线的文档。
      2.  仔细阅读检索到的路线文档，**提取其中的 'boatId'。**
      3.  **然后，再次调用 `retrieve_boats` 工具，这次使用提取到的 'boatId' 作为过滤参数，并以“船只详细信息”或“船只名称”等作为查询，来获取船只的详细文档。**
      4.  将两次检索到的信息综合起来，形成完整的答案。

    **多步推理示例 (请学习并灵活运用):**
    *   **用户问：“南极路线的船有哪些？”**
        1.  **Thought:** 用户想知道路线的船名，这是船宿类问题。我需要先调用 `retrieve_tours` 工具并过滤 `category` 来检索路线信息，获取 `boatId`，然后用 `boatId` 查找船只信息。
        2.  **Action:** 调用 `retrieve_tours(query="南极路线", locationName="南极", category=["船宿船舶信息", "船宿路线", "船宿行程"])`
        3.  **Observation:** 检索到路线文档，其中提到 `boatId` 为 `ABC-001`。
        4.  **Thought:** 我已获得 `boatId`，现在需要用它来查找船只名称。
        5.  **Action:** 调用 `retrieve_boats(query="船只名称", boatId="ABC-001", category=["船宿船舶信息", "船宿路线", "船宿行程"])`
        6.  **Observation:** 检索到船只`ABC-001`的文档，其中船名为“探险号”。
        7.  **Thought:** 我已获得所有信息，可以回答用户的问题。
        8.  **Final Answer:** 南极路线的船有“探险号”。

    **最终判断：**
    *   如果知识库中没有相关信息，请明确说明。

    注意：请用流畅的中文自然语言回答，不要使用 Markdown 或其他格式，尤其不要出现星号（*），保持纯文本。
    """

    def create_rag_agent_with_memory() -> Runnable:
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。请在 .env 文件中配置。")

        # 工具列表包括所有新的专用检索工具
        tools = [retrieve_tours, retrieve_trips, retrieve_boats, retrieve_general_knowledge]

        agent = create_agent(
            agent_llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            system_prompt=AGENT_PROMPT_STR,  # <-- 传入渲染后的字符串
        )
        return agent

    return create_rag_agent_with_memory()


# 激活 Agent
dive_agent = get_agent()

# --- 定义三种专业 UI 组件 ---
def ui_wiki_card(doc):
    """展示目的地百科，侧重季节、难度和看点"""
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### 🗺️")
            st.markdown(f"**{doc.metadata.get('locationName', '目的地')}**")
        with col2:
            # 使用 Emoji 模拟标签
            difficulty = doc.metadata.get('experience', '未知')
            season = doc.metadata.get('departureMonth', '全年')
            st.markdown(f"**难度:** `{difficulty}` | **最佳季节:** `{season}月`")
            st.markdown(f"**必看生物:** {doc.metadata.get('nameCN', '各种海洋生物')}")

        # 针对用户等级的温馨提示
        user_lv = st.session_state.get('user_level', 'OW')
        if "难" in str(difficulty) and user_lv == "OW":
            st.warning("⚠️ 此地流大，建议考取 AOW 或积累更多瓶数后再前往。")

def ui_trip_card(doc):
    """展示船宿信息，侧重日期、价格、跳转"""
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            name = doc.metadata.get('nameCN', '精品船宿')
            date = doc.metadata.get('departureDate_display', '近期出发')
            st.markdown(f"**🚢 {name}**")
            st.caption(f"📅 出发日期: {date}")
        with c2:
            price = doc.metadata.get('price', '电询')
            st.markdown(f"**💰 {price}**")
            st.caption("起/人")
        with c3:
            # 这里的链接你可以根据你的数据动态生成
            st.link_button("查看详情", "https://cooldive.com", use_container_width=True)

def ui_knowledge_card(doc):
    """展示复习知识点，侧重权威性和教练建议"""
    st.info(f"💡 **划重点**: {doc.page_content[:200]}...")
    with st.expander("📖 查看完整手册说明"):
        st.write(doc.page_content)
        st.caption("来源：专业潜水教学手册")

# --- 根据文档的 category 自动选择组件 ---#
def render_adaptive_ui(docs):
    """
    智能 UI 匹配器：根据元数据特征自动选择模板
    """
    if not docs:
        return

    st.divider()

    # 将 3 个船宿子类定义为一个集合，方便判断
    LIVEABOARD_CATS = {"船宿船舶信息", "船宿路线", "船宿行程"}

    for i, doc in enumerate(docs[:3]):  # 每次最多展示 3 个组件，防止页面太乱
        meta = doc.metadata
        category = meta.get("category", "通用")

        # --- 策略 A：船宿类模板 (精确匹配已知的大类) ---
        if category in LIVEABOARD_CATS or "price" in meta:
            render_trip_card(doc, i)

        # --- 策略 B：百科类模板 (识别特征字段：locationName) ---
        elif "locationName" in meta:
            render_wiki_card(doc, i)

        # --- 策略 C：通用知识模板 (兜底方案) ---
        else:
            render_knowledge_card(doc, i)


# --- 具体的组件实现（更加通用化） ---

def render_trip_card(doc, idx):
    """交易型卡片：突出价格和日期"""
    with st.container(border=True):
        st.caption(f"🚢 {doc.metadata.get('category', '船宿信息')}")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{doc.metadata.get('nameCN', '未命名航线')}**")
            date = doc.metadata.get('departureDate_display', '请咨询客服')
            st.markdown(f"📅 出发日期: `{date}`")
        with c2:
            price = doc.metadata.get('price', '电询')
            st.button(f"💰{price}", key=f"trip_{idx}")


def render_wiki_card(doc, idx):
    """百科型卡片：展示标签云"""
    with st.container(border=True):
        st.caption(f"🗺️ {doc.metadata.get('category', '目的地百科')}")
        st.markdown(f"#### {doc.metadata.get('locationName', '未名地点')}")

        # 动态提取所有元数据作为标签展示 (去除掉已知的长字段)
        tags = []
        for k in ["experience", "departureMonth", "rating", "dives"]:
            if k in doc.metadata:
                tags.append(f"#{doc.metadata[k]}")

        if tags:
            st.markdown(" ".join([f"`{t}`" for t in tags]))
        st.write(f"{doc.page_content[:100]}...")


def render_knowledge_card(doc, idx):
    """复习型卡片：重点展示文字内容"""
    with st.chat_message("ai", avatar="💡"):
        st.caption(f"📚 知识点: {doc.metadata.get('category', '潜水百科')}")
        st.markdown(doc.page_content)
        if "source" in doc.metadata:
            st.caption(f"来源: {doc.metadata['source']}")

# --- 4. Streamlit 界面逻辑 ---
# --- 侧边栏：潜水员档案 ---
with st.sidebar:
    st.header("🤿 我的潜水档案")
    st.caption("AI 将根据你的档案给出个性化建议")

    # 1. 等级选择
    dive_level = st.selectbox(
        "潜水等级",
        ["初学者 (无证)", "OW (开放水域)", "AOW (进阶开放水域)", "Rescue (救援员)", "DM/教练"],
        index=2  # 默认选 AOW
    )

    # 2. 瓶数输入
    dive_logs = st.number_input("总潜水瓶数 (Logs)", min_value=0, value=50, step=1)

    # 3. 偏好选择
    # 先定义选项
    pref_options = ["大货 (鲨鱼/Manta)", "微距 (海兔/小虾)", "放流", "沉船", "洞穴", "水下摄影"]

    interests = st.multiselect(
        "潜水偏好",
        options=pref_options,
        default=[pref_options[0]]  # 自动选第一项
    )

    st.divider()

    # 4. 这里的状态会存入 session_state
    user_profile = f"""
    - 等级: {dive_level}
    - 经验: {dive_logs} 瓶
    - 偏好: {", ".join(interests)}
    """

    # 展示当前档案（可选，方便调试）
    if st.checkbox("显示 AI 感知的档案"):
        st.text(user_profile)

st.title("🤿 DiveMind AI 潜水 Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("问我关于潜水行程、船宿或知识点..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 调用 Agent 获取回答 ---
    with st.chat_message("assistant"):
        # 1. 在这里实时获取侧边栏的最新值
        # 构造一个强力的、带有用户背景的“指令前缀”
        context_prefix = f"""【当前访客档案 - 必须作为建议的依据】
    - 潜水等级：{dive_level}
    - 潜水经验：{dive_logs} 瓶
    - 兴趣偏好：{", ".join(interests)}
    ---
    """

        # 2. 构造输入数据
        # 我们把背景信息直接拼在用户问题的最前面，这是最强力的注入方式
        input_data = {
            "messages": [
                HumanMessage(content=f"{context_prefix}\n请基于我的档案回答：{prompt}")
            ]
        }

        # 3. 执行调用
        config = {"configurable": {"thread_id": "diver_user_1"}}

        # 开启一个加载动画，增加专业感
        with st.spinner("正在基于你的潜水档案生成建议..."):
            try:
                result = dive_agent.invoke(input_data, config)
            except Exception as e:
                st.error(f"❌ 运行出错：{str(e)}")
                # 这行会在控制台打印完整的错误，方便你在 Manage app 里的 logs 查看
                print(f"ERROR DETAILS: {e}")
                st.stop()

        # 4. 提取回答
        final_answer = result["messages"][-1].content
        st.markdown(final_answer)

        # --- 【增加这一行调试代码】 ---
        st.write("DEBUG：篮子里现在的文档数量是：", len(st.session_state.get("last_retrieved_docs", [])))
        # ---------------------------

        if st.session_state.get("last_retrieved_docs"):
            render_adaptive_ui(st.session_state.last_retrieved_docs)
            st.session_state.last_retrieved_docs = []

        # 存入聊天记录
        st.session_state.messages.append({"role": "assistant", "content": final_answer})



