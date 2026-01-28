# config.py
import os
from pathlib import Path

# =======================================================
# 1. 环境自感知与模式切换 (Environment & Mode)
# =======================================================
# 手动控制开关：可选 "LOCAL" 或 "CLOUD"
# 建议：本地跑 ingest 用 "LOCAL" 或 "CLOUD" 按需切换；云端 App 运行时建议设为 "CLOUD"
EMBEDDING_MODE = "CLOUD"

# =======================================================
# 2. 路径锚定 (Project Root Anchor)
# =======================================================
# 因为 config.py 就在根目录，所以直接获取当前文件所在目录即可
ROOT_DIR = Path(__file__).resolve().parent

# =======================================================
# 3. 数据中心路径 (Data Hierarchy)
# =======================================================
DATA_DIR = ROOT_DIR / "data"

# --- 原始数据层 (Raw) ---
RAW_DATA_DIR = DATA_DIR / "raw" / "liveaboard_crawler"
RAW_DATA_A = RAW_DATA_DIR / "cooldive_boat_details_with_trips_1-10.json"
CRAWLER_STATE_FILE = RAW_DATA_DIR / "crawled_boat_ids.json"

# --- 清洗数据层 (Processed) ---
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"

# --- 用户记忆存储路径 ---

USER_MEMORY_DIR = DATA_DIR / "user_memory"

# =======================================================
# 4. 向量数据库 (Vector DB Path Logic)
# =======================================================
# 数据库版本控制
ACTIVE_DB_VERSION = "v3"  # <-- 在这里修改版本号

# --- 配置模式选择 ---
# 设置为 True 进行增量更新 (只处理修改过的文件和新文件)
# 设置为 False 进行全量重建 (清空数据库，从头开始)
INCREMENTAL_UPDATE_DEFAULT = False # <-- 在这里修改 True 或 False 来切换模式

if EMBEDDING_MODE == "LOCAL":
    # 指向本地模型生成的索引
    CHROMA_PATH = str(DATA_DIR / "db_local" / "local_model" / "bge-m3" / ACTIVE_DB_VERSION)
else:
    # 指向硅基流动 API 生成的索引
    CHROMA_PATH = str(DATA_DIR / "db_cloud" / "sili" / "BAAIbge-m3" / ACTIVE_DB_VERSION)

# 增量更新状态文件路径 (与数据库版本绑定)
DOC_STATUS_FILE = os.path.join(CHROMA_PATH, "document_status.json")

# --- 业务元数据配置 ---
DEFAULT_PROJECT_NAME = "DiveKnowledgeBase"
MARKDOWN_DOC_VERSION = "1.0"

# --- 外部数据文件 ---
# 假设这个 Excel 放在 data/processed 目录下
EXTERNAL_METADATA_EXCEL = DATA_DIR / "processed_data" / "chunks_with_category.xlsx"
EXTERNAL_METADATA_CHUNK_ID_COL = "Chunk_ID_Hash"
EXTERNAL_METADATA_CATEGORY_COL = "Category"

# --- 知识库源文件夹 (Markdown/PDF) ---
KNOWLEDGE_BASE_DIR = DATA_DIR / "docs"

# =======================================================
# 5. 模型详细配置 (Model Config)
# =======================================================
# --- 云端 API 配置 ---
SILICON_BASE_URL = "https://api.siliconflow.cn/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

EMBEDDING_MODEL_CLOUD = "BAAI/bge-m3"
# 负责意图识别、实体抽取、query重写的小模型，目前从硅基流动接入
SLM_MODEL = "Qwen/Qwen3-8B"
# 主agent大模型
RAG_LLM_MODEL = "deepseek-chat"

# --- 本地模型配置 (仅脚本本地运行时有效) ---
LOCAL_BGE_PATH = Path("E:/Python项目/dify应用的评估效果/local_bge_m3_model/bge-m3")

# =======================================================
# 7. 提示词 (Prompts)
# =======================================================

# --- 小模型提示词 ---
INTENT_PARSER_PROMPT = """你是一个潜水领域的语义解析专家。
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
1. 意图识别（intent）：
    intent只能从"CHITCHAT" 和 "CONSULT" 中选一个，不能乱写。如果只是闲聊分享（如“看到海龟了”），intent 设置为 CHITCHAT。
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

    
   
# 严格约束 (Strict Constraints)
2. 所有的值，如果是字符串或范围（如 7-10），必须加双引号。
4. keywords 列表绝对不能为空，至少要包含用户提到的地点。
"""

# --- 记忆提取提示词 ---
MEMORY_EXTRACTION_PROMPT = """
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

# --- 主agent大模型DeepSeek 回复生成提示词 ---

BUDDY_RESPONSE_PROMPT = """
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

# =======================================================
# 8. 自动化目录检查
# =======================================================
# 确保程序运行时，必要的文件夹都已经存在，不会报错
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, USER_MEMORY_DIR, Path(CHROMA_PATH)]:
    os.makedirs(folder, exist_ok=True)