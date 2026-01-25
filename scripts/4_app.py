try:
    # 这一段是给云端环境准备的补丁
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # 如果在本地运行，找不到 pysqlite3 也没关系，直接跳过
    pass

import streamlit as st
st.set_page_config(page_title="DiveMind AI", page_icon="🤿")
import uuid
import os
import json
import re
import jieba
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool


import datetime
import calendar
from typing import Optional, List, Dict, Any, Tuple


from langchain_community.retrievers import BM25Retriever

load_dotenv()

# RAG 系统配置
# =======================================================
# 知识库（后缀为embedding模型）
CHROMA_PATH = "chroma_by_sili_bge-m3"
# 主agent大模型
RAG_LLM_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# 负责意图识别、实体抽取、query重写的小模型，目前从硅基流动接入
SLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SILICON_BASE_URL = "https://api.siliconflow.cn/v1"
# embedding模型（query向量化），目前从硅基流动接入
EMBEDDING_MODEL = "BAAI/bge-m3"


# 用户记忆模块
# =======================================================
# 存储路径配置（建议使用绝对路径，确保文件一定能找到）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, "user_memory")
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)


# --- 识别逻辑：硬件指纹版 ---
def get_persistent_user_id():
    # 方案 A：获取电脑节点 ID（只要不换主板/网卡，这个值是唯一的）
    # 这种方式在本地调试非常稳
    node_id = str(uuid.getnode())

    return f"diver_{node_id[:8]}"


user_id = get_persistent_user_id()
user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")

# --- 确保加载逻辑准确 ---
if "user_profile" not in st.session_state:
    if os.path.exists(user_file):
        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                st.session_state.user_profile = json.load(f)
                st.session_state.onboarding_complete = True
                print(f">>> 成功识别老用户: {user_id}")
        except Exception as e:
            print(f">>> 读取文件失败: {e}")
            st.session_state.onboarding_complete = False
    else:
        print(f">>> 识别为新用户: {user_id}，正在创建档案")
        st.session_state.user_profile = {"level": None, "logs": None, "preference": []}
        st.session_state.onboarding_complete = False
        st.session_state.onboarding_step = 1


def extract_new_memory(user_input, ai_response):
    """
    这是一个静默运行的函数，专门用来提取对话中的偏好和生物信息。
    """
    profile = st.session_state.user_profile

    # 构造一个极简的提示词
    system_prompt = """
你是一个潜水教练助理。请分析对话，提取新信息并严格以JSON格式返回。

【重要：区分意向与事实】
- 只有当用户明确表示“去过”、“在那潜过”、“上次在XX”、“见过”时，才存入。
- 如果用户只是说“想去”、“计划去”、“打算去”、“想看”，严禁存入。

1. new_animals: 见过的生物
2. new_divesites: 【确实去过】的地点 (严禁包含计划想去的地点)
3. new_prefs: 偏好或痛点
4. new_tips: 个人经验或心得

必须返回格式：{"new_animals": [], "new_prefs": [], "new_divesites": [], "new_tips": []}
    """

    user_context = f"用户说: {user_input}\n教练说: {ai_response}"

    try:
        response = agent_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_context)
        ])

        # --- 健壮的 JSON 解析 ---
        content = response.content.strip()
        # 移除 AI 可能包裹的 ```json 标签
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()

        new_info = json.loads(content)
        updated = False

        # 定义一个简单的映射关系，把 JSON 里的键存入 profile
        mapping = {
            "new_animals": "seen_animals",
            "new_divesites": "visited_sites",
            "new_prefs": "dynamic_notes",
            "new_tips": "dive_tips"
        }

        for json_key, profile_key in mapping.items():
            items = new_info.get(json_key, [])
            if items:
                if profile_key not in profile: profile[profile_key] = []
                for item in items:
                    if item not in profile[profile_key]:
                        profile[profile_key].append(item)
                        updated = True

        if updated:
            # 存入硬盘
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=4)
            # 在黑窗口打印一下，方便确认
            print(f">>> 记忆库已成功更新：{new_info}")
            return True

    except Exception as e:
        print(f"记忆提取出错: {e}")
    return False
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
# 初始化 RAG 组件 (LLM 、SLM、embedding和 ChromaDB)
# =======================================================


agent_llm = ChatOpenAI(
    model=RAG_LLM_MODEL,
    openai_api_base=DEEPSEEK_BASE_URL,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.0
)
if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。请在 .env 文件中配置。")

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=SILICON_BASE_URL
)

slm_parser = ChatOpenAI(
    model=SLM_MODEL,
    api_key=st.secrets["SILICONFLOW_API_KEY"],
    base_url=SILICON_BASE_URL,
    temperature=0.1 # 越低越稳定
)

if not os.path.exists(CHROMA_PATH):
    raise ValueError(f"ChromaDB 路径 '{CHROMA_PATH}' 不存在。请确保已运行 ingest.py 创建了知识库。")
rag_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


# =======================================================
# 定义 Agent 的专用检索工具
# =======================================================
# 定义分词函数
def chinese_tokenizer(text):
    # 过滤掉一些没用的标点符号，只留词项
    return [word for word in jieba.lcut(text) if len(word.strip()) > 0]

def calculate_buffered_range(year, month):
    """
    根据年、月计算带 8 小时冗余的 Unix 时间戳范围
    """
    # 1. 目标月份的第一天 00:00:00
    start_dt = datetime.datetime(year, month, 1, 0, 0, 0)
    # 往前推 8 小时 (时区冗余)
    start_timestamp = start_dt.timestamp() - (8 * 3600)

    # 2. 目标月份的最后一天 (例如 1月31日)
    last_day = calendar.monthrange(year, month)[1]
    # 用户要求：扩展到“次月1日的 23:59:59”
    # 逻辑：目标月最后一天 + 1天 = 次月1日
    end_dt = datetime.datetime(year, month, last_day, 23, 59, 59) + datetime.timedelta(days=1)
    # 往后延 8 小时 (时区冗余)
    end_timestamp = end_dt.timestamp() + (8 * 3600)

    return start_timestamp, end_timestamp

def parse_trip_content(content):
    """
    使用正则表达式从category为“船宿行程”文件， page_content 字符串中提取结构化字段
    """
    details = {}

    # 定义提取规则 (匹配 【】 内部的内容)
    patterns = {
        "tripId": r"旅程ID: 【(.*?)】",
        "tourId": r"所属路线ID: 【(.*?)】",
        "boatId": r"所属船只ID: 【(.*?)】",
        "nameCN": r"旅程名称: 【(.*?)】",
        "nitrox": r"高氧供应: 【(.*?)】",
        "wifi": r"Wi-Fi供应: 【(.*?)】",
        "tech_friendly": r"技术潜水友好: 【(.*?)】",
        "dives": r"潜水次数: 【(.*?)】",
        "duration": r"行程天数: 【(.*?)】",
        "nights": r"晚数: 【(.*?)】",
        "updatedTime": r"最后更新时间: 【(.*?)】",
        "booking_policy": r"该旅程的预订政策: 【(.*?)】",
        "cancellation_policy": r"该旅程的取消政策: 【(.*?)】",
        "available_count": r"实时空位\(非实时\): 【(.*?)】",
        "price_str": r"实时价格\(非实时\): 【(.*?)】"
    }

    # 1. 执行正则匹配
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        details[key] = match.group(1) if match else "未知"

    # 2. 特殊处理：提取括号里的英文名
    # 格式：旅程名称: 【中文名】(英文名)
    en_name_match = re.search(r"旅程名称: 【.*?】\((.*?)\)", content)
    details["nameEN"] = en_name_match.group(1) if en_name_match else ""

    # 3. 数据类型二次清洗 (可选)
    # 如果价格包含 ￥，可以去掉只留数字方便后续计算
    if details["price_str"] != "未知":
        details["price_value"] = details["price_str"].replace("￥", "").replace("元", "").strip()

    return details

def parse_user_intent(query, chat_history):
    """
    调用小模型进行意图识别、实体抽取、query重写，返回结构化 JSON
    """
    # --- 【获取系统当前时间】 ---
    now = datetime.datetime.now()
    current_date_str = now.strftime("%Y年%m月%d日")
    current_month = now.month
    current_year = now.year

    system_prompt = """你是一个潜水领域的语义解析专家。
【当前时间：{current_date_str}】
【当前访客侧边栏档案：{user_profile_sidebar}】

任务：分析用户提问，提取检索参数。
必须严格返回 JSON，严禁开场白。格式如下：
{
  "intent": "CHITCHAT" 或 "CONSULT",
  "topic": "TRIP" 或 "KNOWLEDGE" 或 "NONE",
  "keywords": ["地点/船名的中英文", "生物名", "装备名", "等级名"],
  "params": {{
     "location_list": ["地点", "地点别名1", "地点别名2", "地点英文名"], 
     "departureYear": 数字,
     "departureMonth": 1-12之间的1个数字,
     "certification": "OW/AOW之一",
     "experience": 数字(瓶数),
     "nitrox": true/false,
     "wifi": true/false,
     "tech_friendly": true/false,
     "is_round_trip": true/false  
  }},
  "search_query": "去掉废话后的核心语义搜索词"
}

【Keywords vs Params 隔离准则】：
Keywords 仅存放：地点名、船名、生物名、感受（如：四王岛、Manta、怕冷）。
Params 仅存放：数字、等级、月份（如：50、AOW、1）。
严禁将数字和等级简写（如 50, OW）放入 keywords 列表，它们会干扰搜索。
地点必须双入：地点名必须同时出现在 keywords 和 params['locationName'] 中。

提取规则：
2. 只要涉及地点，必须在 location_list 中通过你的常识进行扩展。
    - 示例：用户说“南极”，输出 "location_list":["南极", "南极洲", "Antarctica"]
    - 示例：用户说“四王岛”，输出 "location_list":["四王岛", "四王群岛", "Raja Ampat"]
3. 参数(params)：
   - 仅当用户在【提问中】明确提到不同于侧边栏的要求时才提取。
   - 设施要求（如“要高氧”）设为 true。
   - 证书和经验：仅提取数字和标准缩写。
   - 严禁脑补：除非用户明确说出“我要高氧”、“有 Wi-Fi 吗”、“需要租装备”，否则 params 里的 nitrox, wifi, tech_friendly 默认设为 null。
   - 默认行为：不要假设船宿一定提供这些设施，你不负责推测，只负责提取。
4 .往返意图 (is_round_trip)：
   - 如果用户只问“X月去”、“X月的行程”，is_round_trip 为 false。
   - 如果用户明确提到“X月往返”、“X月回来”、“X月结束”，is_round_trip 为 true
5.意图识别（intent）：
   -如果只是闲聊分享（如“看到海龟了”），intent 设置为 CHITCHAT。
    
   
# 严格约束 (Strict Constraints)
2. 所有的值，如果是字符串或范围（如 7-10），必须加双引号。
3. 严禁输出 2023 年，当前是 2026 年！
4. keywords 列表绝对不能为空，至少要包含用户提到的地点。



"""

    try:
        response = slm_parser.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"对话上下文：{chat_history[-2:]}\n用户当前输入：{query}")
        ])
        print(f"DEBUG: 7B 小模型原始回答内容: {response.content}")
        # 清洗 JSON
        content = response.content.strip()
        json_str = re.search(r'\{.*\}', content, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return {"intent": "CONSULT", "topic": "NONE", "keywords": [], "params": {}, "search_query": query}
    print(f"7b返回内容：")
    print(response)

def get_correct_year_and_month(extracted_month, extracted_year=None):
    """
    核心逻辑：根据当前时间自动修正年份(避免AI判断时间出错）
    """
    now = datetime.datetime.now()
    curr_year = now.year    # 2026
    curr_month = now.month  # 1 (假设今天是1月14日)

    # 如果用户没说年份，或者 AI 乱给了一个过去的年份（比如 2023）
    if not extracted_year or extracted_year < curr_year:
        # 逻辑判断：
        # 如果提取的月份 >= 当前月份，说明是今年的那个月（或者就是这个月）
        if extracted_month >= curr_month:
            final_year = curr_year
        # 如果提取的月份 < 当前月份，说明是明年的那个月
        else:
            final_year = curr_year + 1
    else:
        # 如果用户明确说了 2027，或者 AI 抓到了未来的年份，则保留
        final_year = extracted_year

    return final_year, extracted_month

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

                # ===针对日期字段的特殊处理 ===
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

    # === 在构建之前，强制锁定 category ===
    # 这样生成的字典会自动包含在 $and 逻辑中，不会产生顶级操作符冲突
    all_params_for_filter["category"] = ["船宿路线"]
    print(f"[工具内部已强制锁定参数 category: ['船宿路线']]")

    # 2. 调用构建函数（使用能自动包 $and 的版本）
    chroma_filters, post_process_filters = _build_filter_dict(**all_params_for_filter)

    print(f"[检索工具将尝试使用以下合法的 ChromaDB 过滤条件: {chroma_filters}]")

    # 3. 使用生成的 chroma_filters 进行检索
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

    # --- 存入全局变量用于UI自动分类 ---
    state_manager.DataStorage.BASKET.extend(final_docs)

    print(f">>>>>> 确认：已存入保险箱，当前大小: {len(state_manager.DataStorage.BASKET)}")
    # -----------------------------------

    return _format_docs(final_docs)


@tool
def retrieve_trips(
        query: str,
        keywords: List[str] = None,  # 接收 7B 提取的关键词
        **kwargs  # 捕获其他未知参数
) -> str:
    """
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
    """
        检索具体的船宿行程信息。
        逻辑：先硬筛选元数据，再对 page_content 执行关键词+向量混合检索。
        """
    print(f"\n[Agent正在执行船宿retrieve_trips检索] Query: '{query}' | 关键词: {keywords}| 其他参数：{kwargs}")
    # 如果 7B 没给 keywords，我们才从 query 里切词，绝对不碰 sidebar 的数字
    final_search_keywords = keywords if keywords else [w for w in jieba.lcut(query) if len(w) > 1]

    # 补充：如果 params 里有地点，地点是可以进入 keywords 的（双入原则）
    loc = kwargs.get("locationName")
    if loc and loc not in final_search_keywords:
        final_search_keywords.append(loc)
    print(f"📍 最终用于内容检索的 Keywords: {keywords}")

    # 2. 构造硬过滤条件 (ChromaDB Filter)
    filter_list = [{"category": {"$eq": "船宿行程"}}]

    # 证书与经验
    if "allowed_levels" in kwargs:
        filter_list.append({"certification": {"$in": kwargs["allowed_levels"]}})
    raw_exp = kwargs.get("max_experience", 0)
    final_exp_val = 0

    try:
        # 如果是字符串且包含横杠 (如 "21-50")
        if isinstance(raw_exp, str) and "-" in raw_exp:
            # 取横杠后面的数字作为上限
            final_exp_val = float(raw_exp.split("-")[-1].strip())
        else:
            # 否则尝试直接转换
            # 使用 re 剔除可能存在的“瓶”字或空格
            import re
            clean_exp = re.sub(r'[^\d.]', '', str(raw_exp))
            final_exp_val = float(clean_exp) if clean_exp else 0
    except Exception as e:
        print(f"⚠️ 经验值转换异常: {raw_exp}, 已重置为0. 错误: {e}")
        final_exp_val = 0

    # 设施筛选 (黑名单法)
    if kwargs.get("needs_nitrox"): filter_list.append({"nitrox": {"$ne": "No"}})
    if kwargs.get("needs_wifi"): filter_list.append({"wifi": {"$ne": "No"}})

    # 时间筛选
    if kwargs.get("departureMonth"):
        year, month = get_correct_year_and_month(kwargs["departureMonth"], kwargs.get("departureYear"))
        start_ts, end_ts = calculate_buffered_range(year, month)
        filter_list.append({"departureDate": {"$gte": start_ts}})
        filter_list.append({"departureDate": {"$lte": end_ts}})

    # 强制转为 int，防止 50.0 导致 ChromaDB 匹配失败
    filter_val = int(float(final_exp_val))
    filter_list.append({"experience": {"$lte": filter_val}})
    print(f"✅ 经验过滤生效：查找要求经验 <= {filter_val} 瓶的行程")

    # if  final_exp_val > 0:
    #     # 强制转为 int，防止 50.0 导致 ChromaDB 匹配失败
    #     filter_list.append({"experience": {"$lte": final_exp_val}})
    #     print(f"✅ 经验过滤生效：查找要求经验 <= {final_exp_val} 瓶的行程")

    chroma_filters = {"$and": filter_list}
    print(f"📍 环节 2 (Metadata 硬过滤): {chroma_filters}")

    # 3. 关键词召回 (BM25) - 只在过滤后的池子里捞
    # 我们直接用 rag_db.get 拿到所有符合硬条件的文档
    all_data = rag_db.get(where=chroma_filters)

    if not all_data or not all_data.get('documents'):
        print("❌ 警告：该硬筛选条件下无数据，检索终止")
        return "Buddy，根据你的等级和时间要求，船宿库里暂时没有匹配的行程哦。"

    # 将数据转为 Document 对象
    candidate_docs = [Document(page_content=t, metadata=m)
                      for t, m in zip(all_data['documents'], all_data['metadatas'])]

    # 4. 执行 BM25 关键词检索 (找字面最相关的)
    bm25 = BM25Retriever.from_documents(candidate_docs, preprocess_func=lambda x: jieba.lcut(x))
    # 我们把 K 设大一点，以便后面进行精细化计分
    bm25.k = 20
    keyword_docs = bm25.invoke(" ".join(keywords) if keywords else query)

    # 5. 计分引擎 (100/N 逻辑)
    scored_list = []
    N = len(keywords) if keywords else 1
    UNIT_SCORE = 100 / N

    print("\n--- 船宿精排计分分析 ---")
    for doc in keyword_docs:
        content_lower = doc.page_content.lower()
        meta = doc.metadata

        total_score = 0
        matched = []

        for kw in keywords:
            kw_l = kw.lower()
            # 权重 A：核心元数据命中 (解决四王岛/四王群岛对齐问题)
            in_meta = kw_l in str(meta.get('tour_nameCN', '')).lower() or \
                      kw_l in str(meta.get('nameCN', '')).lower() or \
                      kw_l in str(meta.get('locationName', '')).lower() or \
                      kw_l in str(meta.get('boat_nameEN', '')).lower()

            # 权重 B：正文 page_content 命中
            in_content = kw_l in content_lower

            if in_meta:
                total_score += UNIT_SCORE
                matched.append(f"{kw}(标题)")
            elif in_content:
                total_score += (UNIT_SCORE * 0.6)  # 正文中了一半分
                matched.append(f"{kw}(正文)")

        # 只要命中了关键词（哪怕一个）就保留
        if total_score > 0:
            scored_list.append({"doc": doc, "score": total_score, "matches": matched})

    # 6. 同船去重 (Unique Boat ID)
    # 按分数从高到低排序
    scored_list.sort(key=lambda x: x["score"], reverse=True)

    winners = []
    seen_boats = set()
    for item in scored_list:
        bid = item["doc"].metadata.get("boatId")
        if bid not in seen_boats:
            winners.append(item)
            seen_boats.add(bid)
        if len(winners) >= 5: break  # 最多给 5 艘船

    # 7. 数据回传与格式化
    state_manager.DataStorage.BASKET.extend([w["doc"] for w in winners])

    print(f">>> 最终选出 {len(winners)} 艘不同的船只资料")

    # 7. 数据回传准备 (保持之前的正则解析逻辑)
    formatted_output = []
    final_docs_for_basket = []

    for i, winner in enumerate(winners):
        doc = winner["doc"]
        parsed_details = parse_trip_content(doc.page_content)
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

    if not winners:
        return "Buddy，没找到匹配的行程。"

    return "\n\n".join(formatted_output)



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

    # --- 存入全局变量用于UI自动分类 ---
    state_manager.DataStorage.BASKET.extend(final_docs)

    print(f">>>>>> 确认：已存入保险箱，当前大小: {len(state_manager.DataStorage.BASKET)}")
    # -----------------------------------

    return _format_docs(final_docs)


# 可以添加一个通用工具用于检索其他类别或无法明确归类的知识
@tool
def retrieve_general_knowledge(query: str, keywords: List[str] = None, **kwargs) -> str:
    """
    带硬核关键词检查的混合检索工具。
    要求：文档必须至少命中一个关键词才会被采纳。
    """
    # --- 【新增：关键词逻辑】 ---
    if not keywords:
        keywords = []
        # 1. 如果 params 里有，把它加进关键词
        for key, val in kwargs.items():
            if val is not None and val != "":
                # --- 【重点修复：月份转换】 ---
                if "departureMonth" in key:
                    # 把 1 变成 "1月"，把 11 变成 "11月"
                    month_str = f"{val}月"
                    if month_str not in keywords:
                        keywords.append(month_str)
                    continue # 处理完月份，跳过后面的通用逻辑

                # 情况 A: 如果值是列表
                if isinstance(val, list):
                    for item in val:
                        item_str = str(item).strip()
                        if item_str and item_str not in keywords:
                            keywords.append(item_str)
                # 情况 B: 如果是普通字符串或数字
                else:
                    val_str = str(val).strip()
                    if val_str and val_str not in keywords:
                        keywords.append(val_str)

    # 2. 如果关键词还是空的，用 jieba 把原始 query 切开当关键词
    if not keywords:
        keywords = [w for w in jieba.lcut(query) if len(w) > 1]

    print(f">>> 最终用于硬匹配的关键词: {keywords}")
    print(f"\n[执行高级检索] 原始提问: {query} | 核心词: {keywords}")

    # 1. 粗排阶段 (Recall)：撒出三张网
    # 第一张网：用户原始提问的向量搜索
    # 第二张网：关键词组合的向量搜索
    # 第三张网：关键词的 BM25 硬匹配搜索

    K_RECALL = 30  # 每张网捞 30 个

    # --- 网 A：原始语义网 ---
    vector_results = rag_db.similarity_search_with_score(query, k=K_RECALL)

    # --- 网 B：关键词语义网 (防止 AI 废话太多干扰语义) ---
    kw_query = " ".join(keywords) if keywords else query
    vector_results_kw = rag_db.similarity_search_with_score(kw_query, k=K_RECALL)

    # --- 网 C：分词后的关键词硬匹配网 (关键修复！) ---
    all_data = rag_db.get()
    all_docs = [Document(page_content=t, metadata=m or {})
                for t, m in zip(all_data['documents'], all_data['metadatas'])]

    # 创建 BM25 时传入中文分词器
    bm25 = BM25Retriever.from_documents(
        all_docs,
        preprocess_func=chinese_tokenizer  # 告诉 BM25 怎么认中文
    )
    bm25.k = K_RECALL
    # 搜索词也要先分词
    tokenized_query = " ".join(chinese_tokenizer(kw_query))
    keyword_docs = bm25.invoke(tokenized_query)

    # 2. 汇总去重
    unique_candidates = {}
    # 合并网 A
    for doc, score in vector_results:
        unique_candidates[doc.page_content[:100]] = {"doc": doc, "v_score": score}
    # 合并网 B
    for doc, score in vector_results_kw:
        content_id = doc.page_content[:100]
        if content_id not in unique_candidates or score < unique_candidates[content_id]["v_score"]:
            unique_candidates[content_id] = {"doc": doc, "v_score": score}
    # 合并网 C
    for doc in keyword_docs:
        content_id = doc.page_content[:100]
        if content_id not in unique_candidates:
            unique_candidates[content_id] = {"doc": doc, "v_score": 1.5}  # 未中向量的给个默认高分

    # 3. 计分与“一票否决”
    scored_list = []
    rejected_list = []
    N = len(keywords) if (keywords and len(keywords) > 0) else 0
    KW_UNIT_SCORE = 100 / N if N > 0 else 0

    print(f"\n--- 正在对 {len(unique_candidates)} 个候选片段进行精排 ---")
    for content_id, item in unique_candidates.items():
        doc = item["doc"]
        v_dist = item["v_score"]
        content_lower = doc.page_content.lower()

        # 只要文档里出现了我们提取的任何一个关键词，就算命中
        matched_words = [kw for kw in keywords if kw.lower() in content_lower] if N > 0 else []

        # --- 【强制：一票否决】 ---
        if N > 0 and not matched_words:
            rejected_list.append({"score": 0, "reason": "未见关键词", "snippet": doc.page_content[:100]})
            continue

            # 计算总分 (关键词权重 100 + 向量补偿 30)
        kw_score = len(matched_words) * KW_UNIT_SCORE
        v_score = max(0, (1.5 - v_dist) / 1.5 * 30)
        total_score = kw_score + v_score

        scored_list.append({
            "doc": doc,
            "total_score": total_score,
            "kw_detail": f"命中 {len(matched_words)}/{N} {matched_words}",
            "v_detail": f"向量分: {v_score:.1f}",
            "snippet": doc.page_content[:200].replace("\n", " ")
        })

    # 4. 排序并择优
    scored_list.sort(key=lambda x: x["total_score"], reverse=True)
    winners = scored_list[:10]

    # --- 控制台获胜者公示 (Console Log) ---
    print("\n🏆 --- 检索精排获胜者 (Top 5) ---")
    if not winners:
        print("❌ 无文档通过硬过滤")
    for i, item in enumerate(winners):
        print(f"排名[{i+1}] 总分: {item['total_score']:.1f} | 理由: {item['kw_detail']} | 片段: {item['snippet'][:50]}...")
    print("--------------------------------\n")

    # 6. 更新保险箱并返回
    state_manager.DataStorage.BASKET.clear()
    state_manager.DataStorage.BASKET.extend([item["doc"] for item in winners])

    if not winners:
        return f"Buddy，我用了三张网都没捞到包含关键词 '{keywords}' 的资料，看来库里还需要补充。"

    return "\n\n".join([f"参考[{i + 1}] (得分:{item['total_score']:.1f}):\n{item['snippet']}"
                        for i, item in enumerate(winners)])


def automated_retrieval_hub(analysis,sidebar_data):
    """
    根据分析结果，自动调度检索工具，填满保险箱 (BASKET)
    """
    state_manager.DataStorage.BASKET.clear()

    if analysis["intent"] == "CHITCHAT":
        return "闲聊模式，无需检索"

    # --- 【调试探针：看看 7B 到底说了什么】 ---
    print(f"\n🔍 [Hub路由诊断]")
    print(f"   - 意图 (intent): {analysis.get('intent')}")
    print(f"   - 主题 (topic): {analysis.get('topic')}")
    print(f"   - 关键词 (keywords): {analysis.get('keywords')}")
    print(f"   - 地点参数: {analysis.get('params', {}).get('location_list')}")

    # 1. 提取基础数据
    params = analysis.get("params", {})
    # 拿到原始关键词
    keywords = analysis.get("keywords", [])

    # --- 【合并地点扩展词】 ---
    # 从 params 里的 location_list 拿出别名
    location_list = params.get("location_list", [])

    # 将别名合并进 keywords，并去重
    # 这样 keywords 就不再只有用户说的那个词，还包含了 AI 扩展出的别名
    if isinstance(location_list, list):
        for loc in location_list:
            if loc not in keywords:
                keywords.append(loc)
    # ----------------------------------

    # 2. 准备硬过滤参数 (这些参数绝对不准进入 keywords)
    user_level = params.get("certification") or sidebar_data.get("level", "OW")
    current_lv = str(user_level).upper()

    if "AOW" in current_lv:
        # 只要包含 AOW（不管有没有+），就给 AOW 的权限
        allowed_levels = ["无证", "OW", "AOW"]
    elif "OW" in current_lv:
        allowed_levels = ["无证", "OW"]
    else:
        # 其他（如无证、初学者等）
        allowed_levels = ["无证"]

    print(f">>> 触发检索逻辑。识别到等级：{user_level} -> 赋予库匹配权限: {allowed_levels}")

    user_logs = params.get("experience") or sidebar_data.get("logs", 0)

    # 3. 构造传递给工具的参数包
    merged_kwargs = {
        "keywords": keywords,  # 现在的 keywords 已经包含了 location_list
        "locationName": params.get("locationName"),  # 保留一个主地点用于可能的后处理
        "departureMonth": params.get("month") or params.get("departureMonth"),
        "departureYear": params.get("year") or params.get("departureYear"),
        "allowed_levels": allowed_levels,
        "max_experience": user_logs,
        "needs_nitrox": params.get("nitrox"),
        "needs_wifi": params.get("wifi")
    }

    topic = analysis["topic"]
    params = analysis["params"]
    keywords = analysis["keywords"]
    query = analysis["search_query"]

    # --- 路由逻辑 ---
    # 只要 params 里有地点，或者 keywords 里有明显的地点词
    has_location = params.get("locationName") or len(keywords) > 0

    # 1. 无论 topic 是什么，只要有地点，就先查百科（环境概况）
    if has_location or topic == "KNOWLEDGE":
        print(">>> 触发百科检索...")
        retrieve_general_knowledge.func(query=query, keywords=analysis["keywords"])

    # 2. 只要有地点，或者明确说要查行程，就查船宿
    if has_location or topic == "TRIP":
        print(f">>> 触发船宿联动检索...用户等级: {user_level}，匹配库内要求: {allowed_levels}")
        # 关键词加上“船宿”二字能让检索更精准
        trip_query = f"{' '.join(keywords)} 船宿"
        retrieve_trips.func(query=trip_query, **merged_kwargs)

    print(f"检索完成，找到 {len(state_manager.DataStorage.BASKET)} 条资料")


def get_response(query, context_docs, user_profile):
    """
    由 DeepSeek 整合最终答案
    """
    # 构造资料块
    doc_text = ""
    if context_docs:
        doc_text = "\n".join([f"[Ref: {i + 1}] {d.page_content[:300]}" for i, d in enumerate(context_docs)])

    final_system_prompt = f"""
    你是 DiveMind，一个超专业的潜伴。
    当前用户信息：{user_profile}

    # 你的任务：
    1. 如果有[参考资料]，你必须优先基于资料回答，并在句末标注 [Ref: n]。
    2. 如果涉及【地点】，即便资料里没写，也要根据常识推荐 PADI 官网链接：
       https://www.padi.com/dive-shops/[国家名英文]/
    3. 严禁展示 ID、Hash 等技术术语。
    4. 语气要像老朋友，如果用户在分享见闻，先热烈回应。
    
    5. 如果参考资料中包含船宿路线或者行程或者船舶，请你在回复中用文字做一个简单的对比。
       - 重点对比：潜水次数、航行日期、价格等等。
       - 示例：你可以说“这几艘船的行程都不错，A船有30次潜水计划，B船有20次潜水计划，如果你体力不错想潜多一些，更推荐去A船”。
    
    6. 引导式回答：
       - 描述行程时，请引用卡片中的数据。
       - 告诉用户：“详细的参数和政策我已经帮你列在下方的卡片里了，你可以点击展开查看原文。”
       
    关于输出格式的硬性约束：
    1.严禁使用 HTML 标签：严禁输出任何 HTML 代码块，包括但不限于 <details>、<summary>、<div>、<br>。
    2.禁止列出资料清单：严禁出现ID等内部数据。严禁在回答结尾生成“参考资料清单”、“来源汇总”或类似的列表。你只需要在对话中通过 [Ref: n] 标注来源。
    3.Markdown 仅限排版：仅允许使用 Markdown 的一级/二级标题（#）、加粗（**）、无序列表（-）来提升内容的可读性。

    """

    input_text = f"【参考资料】：\n{doc_text}\n\n用户问题：{query}"

    # 使用 .stream 替代 .invoke
    for chunk in agent_llm.stream([
        SystemMessage(content=final_system_prompt),
        HumanMessage(content=input_text)
    ]):
        # 这里的 chunk.content 是流出的每一个字符
        yield chunk.content

# ---- 负责从保险箱中提取文档、去重，并调用渲染函数
def display_trip_results(docs):
    if not docs:
        return

    st.write("---")
    st.caption("📑 匹配到的船宿详细方案")

    # --- 【去重逻辑中心】 ---
    seen_boat_ids = set()
    display_count = 0

    for doc in docs:
        # 只要船宿行程的数据
        if doc.metadata.get("category") != "船宿行程":
            continue

        boat_id = doc.metadata.get("boatId")

        # 如果这艘船已经出现过，直接跳过
        if boat_id in seen_boat_ids:
            continue

        # 如果是新船，记录并渲染
        seen_boat_ids.add(boat_id)
        render_super_trip_card(doc, display_count)

        display_count += 1
        # 最多显示 5 艘不同的船，防止页面太长
        if display_count >= 5:
            break
# ----船宿卡片----
def render_super_trip_card(doc, idx):
    """
    15个字段的超级卡片：结构化、图标化、层级分明
    """

    # 1. 调用之前写的正则解析函数
    details = parse_trip_content(doc.page_content)
    # 补充从 metadata 拿到的稳定字段
    meta = doc.metadata

    with st.container(border=True):
        # --- 第一层：头部 (名称与价格) ---
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.markdown(f"### 🚢 {meta.get('boat_nameEN', '未知')}{meta.get('boat_nameCN', '')}")
            st.caption(f" | 📍 {meta.get('locationName', '未知')}")
            st.caption(f" | 路线：{meta.get('tour_nameCN', '未知')}({meta.get('tour_nameEN', '')})")
            st.caption(f" | 出发起点：{meta.get('departureLocation', '未知')}  返程终点：{meta.get('arrivalLocation', '未知')}")
        with col_h2:
            st.markdown(f"### :orange[{details['price_str']}]")
            st.caption(f"🔥 余位: {details['available_count']}")

        st.divider()

        # --- 第二层：核心规格 (4列并排) ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.write("⏱️ **时长**")
            st.write(f"{meta.get('duration', '未知')}天{meta.get('nights', '未知')}晚")
        with c2:
            st.write("🤿 **潜水**")
            st.write(f"{meta.get('dives', '未知')}次")
        with c3:
            st.write("🎖️ **要求**")
            st.write(f"{meta.get('certification', 'AOW')}")
        with c4:
            st.write("📊 **经验**")
            st.write(f"{meta.get('experience', 50)} 瓶")

        # --- 第三层：服务设施 (小标签排布) ---
        # 使用 markdown 模拟小标签
        nitrox_icon = "✅" if details['nitrox'] == "是" else "❌"
        wifi_icon = "✅" if details['wifi'] == "是" else "❌"
        tech_icon = "✅" if details['tech_friendly'] == "是" else "❌"

        st.markdown(
            f"高氧: {nitrox_icon} | Wi-Fi: {wifi_icon} | 技术潜水友好: {tech_icon}"
        )

        # --- 第四层：日期信息 ---
        st.info(f"📅 **航期：** {meta.get('departureDate_display', '未知')}  ➡️  {meta.get('arrivalDate_display', '未知')}")
        # --- 第五层：政策与外链 (折叠显示) ---
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            with st.expander("📝 预订与取消政策 (查看详情)"):
                st.write("**预订政策:**")
                st.caption(details['booking_policy'])
                st.write("**取消政策:**")
                st.caption(details['cancellation_policy'])
                st.warning("💡 提示：政策内容较长，如有疑问可让 Buddy 为你总结。")
        with col_f2:
            # 按钮跳转原站
            source_url = meta.get("Metadata_source", "#")
            st.link_button("🔗 查看原站详情", source_url, use_container_width=True)

# --- 4. Streamlit 界面逻辑 ---

# -----侧边栏用户画像---
with st.sidebar:
    st.header("🤿 潜水备忘录")

    if not st.session_state.onboarding_complete:
        # --- 新手模式：显示进度条 ---
        progress = (st.session_state.onboarding_step - 1) / 3
        st.write("正在建立连接，请在对话框完成初始化...")
        st.progress(progress)
        st.caption(f"进度：{int(progress * 100)}%")
    else:
        # --- 老手模式：展示画像标签 ---
        st.success("✅ buddy已记下你的档案")
        profile = st.session_state.user_profile

        st.write(f"**等级：** {profile['level']}")
        st.write(f"**经验：** {profile['logs']}")

        st.write("**偏好标签：**")
        # 这里的标签支持手动删除
        for i, tag in enumerate(profile['preference']):
            cols = st.columns([4, 1])
            cols[0].caption(f"• {tag}")
            if cols[1].button("❌", key=f"del_{i}"):
                st.session_state.user_profile['preference'].pop(i)
                # 保存修改到文件
                with open(user_file, 'w') as f:
                    json.dump(st.session_state.user_profile, f)
                st.rerun()

    profile = st.session_state.user_profile

    st.subheader("🗺️ 足迹")
    sites = profile.get("visited_sites", [])
    st.write(" ".join([f"`{s}`" for s in sites]) if sites else "还没留下足迹")

    st.subheader("🐬 生物集邮")
    animals = profile.get("seen_animals", [])
    st.write(" ".join([f"`{a}`" for a in animals]) if animals else "还没集邮")

    st.subheader("🧠 教练笔记")
    # 合并展示心得和偏好
    notes = profile.get("dynamic_notes", []) + profile.get("dive_tips", [])
    if notes:
        for n in notes:
            st.info(n)

# -----主界面逻辑---


# --- 1. 初始化 Session State ---
if "review_mode" not in st.session_state:
    st.session_state.review_mode = False  # 默认是聊天模式
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.review_mode:
    # 场景 A：聊天模式
    # ==========================
    st.title("🤿 DiveMind AI Agent")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
else:
    # 场景 B：复习模式
    # ==========================
    import quiz_module # 导入你的新脚本
    quiz_module.render_quiz_page() # 调用副脚本的渲染函数


# --- 主界面逻辑 ---
if not st.session_state.onboarding_complete:
    st.title("🌊 欢迎来到 DiveMind")
    with st.chat_message("assistant", avatar="🤿"):

        if st.session_state.onboarding_step == 1:
            st.write("嗨！我是你的私人潜水buddy。为了给你更好的建议，能告诉我你的**潜水等级**吗？")
            cols = st.columns(3)
            if cols[0].button("初学者/无证"):
                st.session_state.user_profile['level'] = "初学者";
                st.session_state.onboarding_step = 2;
                st.rerun()
            if cols[1].button("OW (开放水域)"):
                st.session_state.user_profile['level'] = "OW";
                st.session_state.onboarding_step = 2;
                st.rerun()
            if cols[2].button("AOW及以上"):
                st.session_state.user_profile['level'] = "AOW及以上";
                st.session_state.onboarding_step = 2;
                st.rerun()

        elif st.session_state.onboarding_step == 2:
            st.write("太棒了！那你的**潜水经验（瓶数）**大概是多少？")
            cols = st.columns(4)
            choices = ["0-20", "21-49", "50-99", "100+"]
            for i, c in enumerate(choices):
                if cols[i].button(c):
                    st.session_state.user_profile['logs'] = c;
                    st.session_state.onboarding_step = 3;
                    st.rerun()

        elif st.session_state.onboarding_step == 3:
            st.write("最后，你最喜欢的**潜水风格**是？")
            choices = ["看大货 (鲨鱼/Manta)", "找微距 (海兔)", "放流潜水", "沉船/洞穴"]
            selected_pref = st.multiselect("可多选", choices)
            if st.button("开启我的潜水之旅"):
                st.session_state.user_profile['preference'] = selected_pref
                st.session_state.onboarding_complete = True

                # 【核心修复】强制写入文件
                try:
                    with open(user_file, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.user_profile, f, ensure_ascii=False, indent=4)
                    st.success(f"档案已保存至本地: {user_id}.json")
                    st.rerun()
                except Exception as e:
                    st.error(f"档案保存失败: {e}")
else:
    # 我们利用一个 container 保证它在输入框上方
    tool_container = st.container()
    with tool_container:
        cols = st.columns([1, 1, 1])  # 预留三个位置，以后可以放地图
        with cols[0]:
            if not st.session_state.review_mode:
                if st.button("🚀 开始行前复习", use_container_width=True):
                    st.session_state.review_mode = True
                    st.rerun()
            else:
                if st.button("🔙 退出复习模式", use_container_width=True):
                    st.session_state.review_mode = False
                    st.rerun()
    # --- 正常对话逻辑 ---
    if prompt := st.chat_input("和你的 Buddy 聊聊..."):
        # 1. 显示用户输入
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Buddy 正在思考...", expanded=True) as status:
                # --- 第一步：小模型解析意图 ---
                status.write("🔍 正在分析你的意图...")
                analysis = parse_user_intent(prompt, st.session_state.messages)

                # --- 第二步：系统自动执行检索 ---
                if analysis["intent"] == "CONSULT":
                    status.write(f"📚 正在为你翻阅知识库: {analysis['keywords']}...")
                    automated_retrieval_hub(analysis,st.session_state.user_profile)
                else:
                    status.write("💬 原来是想找我叙叙旧，这就来！")

                status.update(label="思考完成!正在打字......", state="complete", expanded=False)

            # --- 第三步：大模型生成回复 ---
            import state_manager

            user_profile = st.session_state.user_profile
            response = get_response(prompt, state_manager.DataStorage.BASKET, user_profile)

            full_response = st.write_stream(response)
            final_answer = full_response
            st.markdown(final_answer)

            # --- 渲染超级卡片 ---
            if state_manager.DataStorage.BASKET:
                display_trip_results(state_manager.DataStorage.BASKET)


            # 存入历史记录
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

            # --- 第五步：静默记忆提取 (异步) ---
            extract_new_memory(prompt, final_answer)



