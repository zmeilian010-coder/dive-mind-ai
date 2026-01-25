import os
from pathlib import Path
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables.base import Runnable
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import datetime
import calendar
from typing import Optional, List, Dict, Any, Tuple

load_dotenv()

# =======================================================
# RAG 系统配置
# =======================================================
CHROMA_PATH = "chroma"
LOCAL_BGE_M3_MODEL_PATH = Path("E:/Python项目/dify应用的评估效果/local_bge_m3_model/bge-m3")
RAG_EMBEDDING_MODEL_NAME = str(LOCAL_BGE_M3_MODEL_PATH)
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

embeddings = HuggingFaceEmbeddings(
    model_name=RAG_EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
)

if not os.path.exists(CHROMA_PATH):
    raise ValueError(f"ChromaDB 路径 '{CHROMA_PATH}' 不存在。请确保已运行 ingest.py 创建了知识库。")
rag_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


# =======================================================
# 定义 Agent 的专用检索工具
# =======================================================

def _build_filter_dict(**params) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    """
    根据传入的参数构建ChromaDB的where过滤字典和需要后续处理的过滤字典。

    参数约定：
    - {key}_eq: 相等匹配 (ChromaDB '$eq')
    - {key}_ne: 不等匹配 (ChromaDB '$ne')
    - {key}_gt: 大于 (ChromaDB '$gt')
    - {key}_gte: 大于等于 (ChromaDB '$gte')
    - {key}_lt: 小于 (ChromaDB '$lt')
    - {key}_lte: 小于等于 (ChromaDB '$lte')
    - {key}_in: 列表包含 (ChromaDB '$in')
    - {key}_nin: 列表不包含 (ChromaDB '$nin')
    - locationName: 字符串包含匹配（需要工具内部后处理）
    - departureMonth: 月份范围匹配（自动转换为日期范围）
    - duration_days_gt/gte/lt/lte: 天数时长过滤

    返回:
    - Tuple[Dict[str, Any], Dict[str, Any]]: (chroma_exact_filters, post_process_filters)
      - chroma_exact_filters: 适用于ChromaDB `where` 参数的精确/范围过滤字典。
      - post_process_filters: 适用于工具内部进行后处理的过滤字典（例如locationName的包含匹配）。
    """
    print(f"\n_build_filter_dict 接收到的参数: {params}") # DEBUG
    chroma_filters: Dict[str, Any] = {}
    post_process_filters: Dict[str, Any] = {}
    current_year = datetime.datetime.now().year

    for key, value in params.items():
        if value is None:
            continue
        print(f"处理参数: {key}={value}") # DEBUG

        # --- 特殊处理字符串包含匹配 ---
        if key == "locationName":
            post_process_filters["locationName"] = str(value)
            print(f"  -> locationName 放入 post_process_filters: {post_process_filters}") # DEBUG
            continue

        # --- 处理 category 列表 ($in) ---  <-- 检查这一块是否正确处理
        if key == "category" and isinstance(value, list):
            if value: # 确保列表不为空
                chroma_filters["category"] = {"$in": value}
                print(f"  -> category 放入 chroma_filters: {chroma_filters}") # DEBUG
            continue

        # --- 特殊处理月份范围 (departureMonth) ---
        # 约定：Agent 传入 departureMonth="11"
        if key == "departureMonth":
            try:
                month = int(value)
                if 1 <= month <= 12:
                    # 构建当前年份该月份的日期范围
                    start_date = datetime.datetime(current_year, month, 1, 0, 0, 0)
                    end_date = datetime.datetime(current_year, month, calendar.monthrange(current_year, month)[1], 23,
                                                 59, 59)

                    # 考虑到用户可能想查明年份的，可以考虑更复杂逻辑，但这里先简化为当前年份
                    # 确保元数据中的日期也是ISO格式
                    chroma_filters["departureDate"] = {
                        "$gte": start_date.isoformat(),
                        "$lte": end_date.isoformat()
                    }
                else:
                    print(f"警告: departureMonth 值 '{value}' 无效，必须是1-12的整数。")
            except ValueError:
                print(f"警告: departureMonth 值 '{value}' 无法转换为整数，跳过过滤。")
            continue

        # --- 处理通用操作符后缀 (gt, gte, lt, lte, eq, ne, in, nin) ---
        # 支持数字和日期字段的比较
        op_mapping = {
            '_eq': '$eq', '_ne': '$ne',
            '_gt': '$gt', '_gte': '$gte',
            '_lt': '$lt', '_lte': '$lte',
            '_in': '$in', '_nin': '$nin'
        }

        found_op = False
        for op_suffix, chroma_op in op_mapping.items():
            if key.endswith(op_suffix):
                field_name = key[:-len(op_suffix)]  # 提取实际的元数据字段名

                # 尝试转换为数字或日期，如果失败则保持原样或报错
                processed_value: Any = value
                if field_name in ['duration', 'dives', 'nights', 'rating', 'yearBuilt']:  # 假设这些是数字字段
                    try:
                        processed_value = int(value)  # 尝试转整数
                    except ValueError:
                        try:
                            processed_value = float(value)  # 尝试转浮点数
                        except ValueError:
                            print(f"警告: 字段 '{field_name}' 的值 '{value}' 无法转换为数字，跳过过滤。")
                            found_op = True
                            break
                elif field_name in ['arrivalDate', 'departureDate', 'updatedTime']:  # 假设这些是日期字段
                    # 确保日期值是ISO格式字符串，Agent应该生成这种格式
                    # 如果Agent生成的是"2024-11-01"，直接使用，或者进一步解析确保符合ISO
                    try:
                        datetime.datetime.fromisoformat(str(value))  # 验证格式
                        processed_value = str(value)
                    except ValueError:
                        print(f"警告: 日期字段 '{field_name}' 的值 '{value}' 不是有效的ISO格式，跳过过滤。")
                        found_op = True
                        break

                if field_name not in chroma_filters:
                    chroma_filters[field_name] = {}
                chroma_filters[field_name][chroma_op] = processed_value
                found_op = True
                break

        if found_op:
            continue

        # 如果所有特殊处理和带操作符后缀的都没有匹配，默认进行精确匹配
        # 这也可能是问题所在，如果所有有值的参数都被特殊处理到 post_process_filters 或被忽略了
        if key not in chroma_filters and key not in post_process_filters and key not in ['departureMonth']: # 避免重复处理
            chroma_filters[key] = value
            print(f"  -> {key} 默认精确匹配放入 chroma_filters: {chroma_filters}") # DEBUG

    print(f"最终 _build_filter_dict 返回: chroma_filters={chroma_filters}, post_process_filters={post_process_filters}") # DEBUG
    return chroma_filters, post_process_filters

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


    # 注意：locals() 会包含所有参数，包括那些带有_gt/_lte后缀的。
    # _build_filter_dict 已经设计为能够解析这些参数。
    # 我们需要传递所有相关参数给 _build_filter_dict
    all_params_for_filter = {k: v for k, v in locals().items()
                             if k not in ['query', 'kwargs', 'self'] and v is not None}
    all_params_for_filter.update(kwargs)  # 合并 kwargs

    chroma_filters, post_process_filters = _build_filter_dict(**all_params_for_filter)

    # === 在工具内部强制添加 category 过滤 ===
    # 注意：确保 "category" 字段在 ChromaDB 中存储的值是与此列表精确匹配的
    forced_category_filter_list = ["船宿路线"]  # retrieve_tours 应该只关注“船宿路线”

    # 如果 chroma_filters 已经有 category 过滤，则与当前的强制列表进行交集（或者覆盖）
    # 这里我们选择直接覆盖或添加，因为这个工具就是为“船宿路线”而生的
    if "category" in chroma_filters and isinstance(chroma_filters["category"], dict) and "$in" in chroma_filters[
        "category"]:
        # 如果 Agent 提供了 category 参数，我们可以选择合并或覆盖
        # 这里为了确保严格过滤为 "船宿路线"，我们选择直接覆盖
        chroma_filters["category"]["$in"] = forced_category_filter_list
    else:
        # 如果没有 category 过滤，就直接添加
        chroma_filters["category"] = {"$in": forced_category_filter_list}

    print(f"[工具内部强制添加 category 过滤: {forced_category_filter_list}]")
    print(f"[检索工具将尝试使用以下ChromaDB过滤条件: {chroma_filters}]")
    # ====================================================================

    retriever = rag_db.as_retriever(search_kwargs={"k": 10, "filter": chroma_filters})
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
        print("[retrieve_tours 工具未找到相关路线信息。]")
        return "未找到相关路线信息。"

    final_docs = final_docs[:5]
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


    all_params_for_filter = {k: v for k, v in locals().items()
                             if k not in ['query', 'kwargs', 'self'] and v is not None}
    all_params_for_filter.update(kwargs)  # 合并 kwargs

    chroma_filters, post_process_filters = _build_filter_dict(**all_params_for_filter)

    # === 在工具内部强制添加 category 过滤 ===
    # 注意：确保 "category" 字段在 ChromaDB 中存储的值是与此列表精确匹配的
    forced_category_filter_list = ["船宿行程"]  # retrieve_tours 应该只关注“船宿行程”

    # 如果 chroma_filters 已经有 category 过滤，则与当前的强制列表进行交集（或者覆盖）
    # 这里我们选择直接覆盖或添加，因为这个工具就是为“船宿行程”而生的
    if "category" in chroma_filters and isinstance(chroma_filters["category"], dict) and "$in" in chroma_filters[
        "category"]:
        # 如果 Agent 提供了 category 参数，我们可以选择合并或覆盖
        # 这里为了确保严格过滤为 "船宿行程"，我们选择直接覆盖
        chroma_filters["category"]["$in"] = forced_category_filter_list
    else:
        # 如果没有 category 过滤，就直接添加
        chroma_filters["category"] = {"$in": forced_category_filter_list}

    print(f"[工具内部强制添加 category 过滤: {forced_category_filter_list}]")
    print(f"[检索工具将尝试使用以下ChromaDB过滤条件: {chroma_filters}]")
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

    all_params_for_filter = {k: v for k, v in locals().items()
                             if k not in ['query', 'kwargs', 'self'] and v is not None}
    all_params_for_filter.update(kwargs)  # 合并 kwargs

    chroma_filters, post_process_filters = _build_filter_dict(**all_params_for_filter)

    # === 核心修改：在工具内部强制添加 category 过滤 ===
    forced_category_filter_list = ["船宿船舶信息"]  # retrieve_tours 应该只关注“船宿船舶信息”

    if "category" in chroma_filters and isinstance(chroma_filters["category"], dict) and "$in" in chroma_filters[
        "category"]:
        chroma_filters["category"]["$in"] = forced_category_filter_list
    else:
        chroma_filters["category"] = {"$in": forced_category_filter_list}

    print(f"[工具内部强制添加 category 过滤: {forced_category_filter_list}]")
    print(f"[检索工具将尝试使用以下ChromaDB过滤条件: {chroma_filters}]")
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

    formatted_docs = format_docs(final_docs)
    print(f"[retrieve_general_knowledge 工具返回了 {len(final_docs)} 个文档，内容片段: {formatted_docs[:200]}...]")
    return formatted_docs


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


# =======================================================
# 问答接口 (供外部调用，也用于交互式主循环)
# =======================================================
_last_retrieved_contexts = []


def ask_rag_agent_with_history(agent_executor: Runnable, question: str, session_id: str) -> tuple[str, list[str]]:
    print(f"\n--- Session '{session_id}' 正在处理问题: '{question}' ---")

    inputs = {"messages": [HumanMessage(content=question)]}

    response_messages = agent_executor.invoke(
        inputs,
        config={"configurable": {"thread_id": session_id}}
    )

    if response_messages and isinstance(response_messages["messages"][-1], AIMessage):
        answer = response_messages["messages"][-1].content
    else:
        answer = "抱歉，无法生成答案。"

    global _last_retrieved_contexts
    _last_retrieved_contexts = []  # 重置

    # --- 从 Agent 的响应中解析实际检索到的上下文 ---
    # LangGraph Agent 的响应消息是列表，我们需要从中提取 ToolMessage 的内容
    retrieved_contexts_from_agent_run = []
    if response_messages and "messages" in response_messages:
        for msg in response_messages["messages"]:
            # ToolMessage 包含工具的实际输出
            # 在 LangGraph/LangChain Agent 中，工具的输出通常会以 AIMessage 的形式出现
            # 或者在 AgentExecutor 的内部痕迹中
            # 然而 create_agent 返回的是一个 graph.compiler().compile() 后的 Runnable，
            # 它的 `invoke` 方法返回的 `messages` 列表里，工具的输出会作为 `AIMessage` 的 content。
            # 或者作为 `ToolMessage` (如果LLM返回的是ToolCall而不是FinalAnswer)
            # 这里的逻辑需要更精确的匹配 ToolMessage 或 HumanMessage(name='Tool')

            # 根据你之前的调试，工具输出被识别为 HumanMessage(name="Tool")
            if isinstance(msg, HumanMessage) and msg.name == "Tool":
                retrieved_contexts_from_agent_run.append(msg.content)
            # 检查是否有 AIMessage 的 content 中包含 "Action" 或 "Observation" 的格式
            elif isinstance(msg, AIMessage):
                # LangGraph Agent 的 AIMessage 可能会包含工具输出，
                # 但更常见的是最终答案，或者Action/Observation的中间步骤
                # 如果 retrieve_documents 返回的是 formatted_docs，那它可能直接出现在最终回答前
                # 暂时我们只捕获 HumanMessage(name="Tool") 的输出
                pass

    _last_retrieved_contexts = retrieved_contexts_from_agent_run

    # 如果没有找到工具的实际输出，或者 RAGAS 需要更“原始”的文档块，
    # 我们可以选择回退到基础检索，但这会失去 Agent 过滤的精确性
    if not _last_retrieved_contexts:
        print("[警告: 未能从Agent响应中解析出检索工具的上下文，回退到基础检索以提供给RAGAS。]")
        base_retriever = rag_db.as_retriever(search_kwargs={"k": 10})
        retrieved_for_contexts = base_retriever.invoke(question)
        _last_retrieved_contexts = [doc.page_content for doc in retrieved_for_contexts]

    return answer, _last_retrieved_contexts


# =======================================================
# 交互式问答主循环
# =======================================================
if __name__ == "__main__":
    print("--- 正在初始化 LangGraph RAG Agent 问答助手 ---")

    try:
        # 创建 Agent
        rag_agent_executor = create_rag_agent_with_memory()
        print("LangGraph RAG Agent 初始化完成。开始多轮对话。")
    except ValueError as e:
        print(f"!!! 错误：LangGraph RAG Agent 初始化失败：{e}")
        exit(1)
    except Exception as e:
        print(f"!!! 错误：Agent 初始化时发生未知异常：{e}")
        import traceback

        traceback.print_exc()
        exit(1)

    interactive_session_id = "interactive_user_session"
    print(f"会话ID: '{interactive_session_id}'。")
    print("\n--- 智能问答工具 (多轮对话模式 - LangGraph Agent) ---")
    print("输入 'reset' 清空对话历史，输入 'exit' 退出。")

    while True:
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            print("退出问答工具。")
            break
        elif user_input.lower() == 'reset':
            print("对话历史已清空。")
            # 重置 LangGraph Agent 的记忆 (InMemorySaver)
            # 最简单的方式是重新创建 Agent 实例。
            rag_agent_executor = create_rag_agent_with_memory()
            print("记忆已重置。")
            continue

        try:
            answer, contexts_for_ragas = ask_rag_agent_with_history(
                rag_agent_executor, user_input, interactive_session_id
            )
            print(f"助手: {answer}")
            if contexts_for_ragas:
                print(f"DEBUG: 检索到的上下文 (第一个片段): {contexts_for_ragas[0][:100]}...")
            else:
                print("DEBUG: 未检索到上下文。")

        except Exception as e:
            print(f"!!! 错误：处理问题时发生异常：{e}")
            import traceback

            traceback.print_exc()
            print("请检查 API Key、网络连接或模型配置。")