# utils.py
import re
import jieba
import datetime
import calendar
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document


def chinese_tokenizer(text: str) -> List[str]:
    """中文分词逻辑"""
    if not text:
        return []
    return [word for word in jieba.lcut(text) if len(word.strip()) > 0]


def calculate_buffered_range(year: int, month: int) -> Tuple[float, float]:
    """根据年、月计算带 8 小时冗余的 Unix 时间戳范围"""
    # 1. 目标月份的第一天 00:00:00
    start_dt = datetime.datetime(year, month, 1, 0, 0, 0)
    # 往前推 8 小时 (时区冗余)
    start_timestamp = start_dt.timestamp() - (8 * 3600)

    # 2. 目标月份的最后一天
    last_day = calendar.monthrange(year, month)[1]
    # 扩展到“次月1日的 23:59:59”
    end_dt = datetime.datetime(year, month, last_day, 23, 59, 59) + datetime.timedelta(days=1)
    # 往后延 8 小时 (时区冗余)
    end_timestamp = end_dt.timestamp() + (8 * 3600)

    return start_timestamp, end_timestamp


def get_correct_year_and_month(extracted_month: int, extracted_year: Optional[int] = None) -> Tuple[int, int]:
    """根据当前时间自动修正年份(避免AI判断时间出错）"""
    now = datetime.datetime.now()
    curr_year = now.year
    curr_month = now.month

    # 如果用户没说年份，或者 AI 乱给了一个过去的年份
    if not extracted_year or extracted_year < curr_year:
        if extracted_month >= curr_month:
            final_year = curr_year
        else:
            final_year = curr_year + 1
    else:
        final_year = extracted_year

    return final_year, extracted_month


def parse_trip_content(content: str) -> Dict[str, Any]:
    """使用正则表达式从 page_content 字符串中提取结构化字段"""
    details = {}
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

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        details[key] = match.group(1) if match else "未知"

    # 提取英文名
    en_name_match = re.search(r"旅程名称: 【.*?】\((.*?)\)", content)
    details["nameEN"] = en_name_match.group(1) if en_name_match else ""

    # 价格数字清洗
    if details.get("price_str") != "未知":
        details["price_value"] = re.sub(r'[^\d.]', '', details["price_str"])

    return details


def _build_filter_dict(**params) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """构建数据库硬筛选条件"""
    print(f"\n_build_filter_dict 接收到的参数: {params}")

    filter_list = []
    post_process_filters = {}
    current_year = datetime.datetime.now().year

    for key, value in params.items():
        if value is None: continue

        if key == "locationName":
            post_process_filters["locationName"] = str(value)
            continue

        if key == "category" and isinstance(value, list):
            if value:
                filter_list.append({"category": {"$in": value}})
            continue

        if key == "departureMonth":
            try:
                month = int(value)
                if 1 <= month <= 12:
                    start_dt = datetime.datetime(current_year, month, 1, 0, 0, 0)
                    last_day = calendar.monthrange(current_year, month)[1]
                    end_dt = datetime.datetime(current_year, month, last_day, 23, 59, 59)
                    filter_list.append({"departureDate": {"$gte": start_dt.timestamp()}})
                    filter_list.append({"departureDate": {"$lte": end_dt.timestamp()}})
            except:
                pass
            continue

        op_mapping = {
            '_eq': '$eq', '_ne': '$ne', '_gt': '$gt', '_gte': '$gte',
            '_lt': '$lt', '_lte': '$lte', '_in': '$in', '_nin': '$nin'
        }
        found_op = False
        for op_suffix, chroma_op in op_mapping.items():
            if key.endswith(op_suffix):
                field_name = key[:-len(op_suffix)]
                processed_value = value
                if field_name in ['departureDate', 'arrivalDate']:
                    try:
                        str_val = str(value).replace("Z", "")
                        processed_value = datetime.datetime.fromisoformat(str_val).timestamp()
                    except:
                        pass
                elif field_name in ['duration', 'dives', 'nights', 'rating']:
                    try:
                        processed_value = float(value)
                    except:
                        pass

                filter_list.append({field_name: {chroma_op: processed_value}})
                found_op = True
                break

        if not found_op and key not in post_process_filters and key != 'departureMonth':
            filter_list.append({key: value})

    final_where = {}
    if len(filter_list) == 1:
        final_where = filter_list[0]
    elif len(filter_list) > 1:
        final_where = {"$and": filter_list}

    return final_where, post_process_filters


def _format_docs(docs: List[Document]) -> str:
    """格式化文档输出给 LLM"""
    if not docs:
        return "未找到相关信息。"
    formatted_list = []
    for i, doc in enumerate(docs):
        content = doc.page_content.replace('\n', ' ').strip()
        metadata_display = {k: v for k, v in doc.metadata.items()
                            if
                            k not in ['source', 'timestamp', 'file_type', 'project', 'processed_by', 'original_source']}
        metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata_display.items()])
        formatted_list.append(f"文档 {i + 1}:\n内容: {content}\n元数据: {metadata_str}\n---")
    return "\n".join(formatted_list)