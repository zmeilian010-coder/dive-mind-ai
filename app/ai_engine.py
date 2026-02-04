# ai_engine.py
import json
import re
import datetime
from langchain_core.messages import HumanMessage, SystemMessage
import models
import state_manager
import tools
import requests
import base64
import json
import uuid
import time
import streamlit as st
import config

# 获取模型实例
slm_parser, agent_llm, _ = models.get_models()


def speech_to_text(audio_bytes):
    """
    使用火山引擎 AUC/SeedASR 大模型进行语音识别
    逻辑：Base64编码 -> Submit提交 -> Query轮询 -> 返回文本
    """
    if not audio_bytes:
        print("❌ 错误：未收到音频数据")
        return None

    print(f"🎤 收到音频，大小: {len(audio_bytes)} 字节")

    # 1. 音频数据转 Base64 字符串
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    # 2. 准备通用的请求头 (Header)
    # 火山引擎的鉴权信息放在 Header 中
    common_headers = {
        "Content-Type": "application/json",
        "X-Api-App-Key": config.VOLC_APP_ID,
        "X-Api-Access-Key": config.VOLC_ACCESS_TOKEN,
        "X-Api-Resource-Id": config.VOLC_CLUSTER,  # 资源ID：volc.seedasr.auc
        "X-Api-Request-Id": str(uuid.uuid4()),  # 每一请求生成的唯一ID
        "X-Api-Sequence": "-1"  # 固定值
    }

    # 3. 构造提交任务 (Submit) 的 Payload
    submit_payload = {
        "app": {
            "appid": config.VOLC_APP_ID,
            "token": config.VOLC_ACCESS_TOKEN,
            "cluster": config.VOLC_CLUSTER
        },
        "user": {"uid": "buddy_user_default"},
        "audio": {
            "format": "wav",
            "data": audio_base64
        }
        # 【核心修改】：直接删掉 additions 字段，或者按下方注释方式写
    }

    try:
        # --- 第一步：提交识别任务 ---
        print(f"📡 正在提交任务至火山引擎...")
        resp = requests.post(config.VOLC_SUBMIT_URL, headers=common_headers, json=submit_payload)
        submit_data = resp.json()

        # 【核心修复】：直接比对数字 1000
        resp_obj = submit_data.get("resp", {})
        if resp_obj.get("code") != 1000:  # 👈 去掉引号，改为数字
            print(f"❌ 任务提交真的失败了: {submit_data}")
            return None

        task_id = resp_obj["id"]
        print(f"✅ 任务提交成功！ID: {task_id}")

        # --- 第二步：循环轮询结果 ---
        # --- 第二步：循环轮询结果 ---
        max_retries = 6
        for i in range(max_retries):
            time.sleep(1)

            # 【核心修改】：将 appid 和 token 全部平铺在最外层
            query_payload = {
                "appid": config.VOLC_APP_ID,  # 账户 ID
                "token": config.VOLC_ACCESS_TOKEN,  # 鉴权 Token
                "id": task_id  # 任务 ID
            }

            print(f"🔍 正在查询结果 ({i + 1}/{max_retries})...")

            # Header 保持不变（包含 X-Api-App-Key 等）
            query_resp = requests.post(
                config.VOLC_QUERY_URL,
                headers=common_headers,
                json=query_payload
            )
            query_data = query_resp.json()

            # --- 结果解析 ---
            q_resp_obj = query_data.get("resp", {})
            # 兼容处理：code 可能是 int 也可能是 str
            q_code = str(q_resp_obj.get("code", ""))

            if q_code == "1000":  # 成功
                final_text = q_resp_obj.get("text", "")
                if final_text:
                    print(f"🎉 识别成功: {final_text}")
                    return final_text
                else:
                    # 如果 code 是 1000 但 text 为空，说明还没写完，继续等
                    print("⏳ 状态 1000 但文字生成中...")
                    continue

            elif q_code == "1001":  # 任务处理中
                print("⏳ 任务处理中 (1001)...")
                continue
            else:
                # 打印出完整的错误 JSON，方便继续调试
                print(f"❌ 查询返回业务错误: {query_data}")
                break

        print("⏳ 语音识别超时")
        return None

    except Exception as e:
        print(f"❌ 调用火山 ASR 发生异常: {e}")
        return None

def parse_user_intent(query, chat_history, user_profile_sidebar):
    """调用 7B 解析意图和参数"""
    now = datetime.datetime.now()
    current_date_str = now.strftime("%Y年%m月%d日")

    # 从 config 载入 prompt 并注入动态变量
    sys_prompt = config.INTENT_PARSER_PROMPT.replace("{current_date_str}", current_date_str)
    sys_prompt = sys_prompt.replace("{user_profile_sidebar}", str(user_profile_sidebar))

    try:
        response = slm_parser.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"对话上下文：{chat_history[-2:]}\n用户输入：{query}")
        ])
        print(f"DEBUG: 7B 小模型原始回答内容: {response.content}")
        content = response.content.strip()
        # JSON 提取与清洗
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"7B解析异常: {e}")

    return {"intent": "CONSULT", "topic": "NONE", "keywords": [], "params": {}, "search_query": query}


def automated_retrieval_hub(analysis, sidebar_data):
    """调度工具库，填充保险箱"""
    state_manager.DataStorage.BASKET.clear()

    if analysis.get("intent") == "CHITCHAT":
        return "闲聊模式"

    # --- 【调试探针：看看 7B 到底说了什么】 ---
    print(f"\n🔍 [Hub路由诊断]")
    print(f"   - 意图 (intent): {analysis.get('intent')}")
    print(f"   - 主题 (topic): {analysis.get('topic')}")
    print(f"   - 关键词 (keywords): {analysis.get('keywords')}")
    print(f"   - 地点参数: {analysis.get('params', {}).get('location_list')}")

    params = analysis.get("params", {})
    keywords = analysis.get("keywords", [])

    # 1. 地点扩展词合并
    location_list = params.get("location_list", [])
    if isinstance(location_list, list):
        for loc in location_list:
            if loc not in keywords: keywords.append(loc)

    # 2. 等级权限处理
    user_level = params.get("certification") or sidebar_data.get("level", "OW")
    current_lv = str(user_level).upper()
    if "AOW" in current_lv:
        allowed_levels = ["无证", "OW", "AOW"]
    elif "OW" in current_lv:
        allowed_levels = ["无证", "OW"]
    else:
        allowed_levels = ["无证"]

    user_logs = params.get("experience") or sidebar_data.get("logs", 0)

    # 3. 构造参数包
    merged_kwargs = {
        "keywords": keywords,
        "locationName": params.get("locationName"),
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
        tools.retrieve_general_knowledge.func(query=query, keywords=analysis["keywords"])

    # 2. 只要有地点，或者明确说要查行程，就查船宿
    if has_location or topic == "TRIP":
        print(f">>> 触发船宿联动检索...用户等级: {user_level}，匹配库内要求: {allowed_levels}")
        # 关键词加上“船宿”二字能让检索更精准
        trip_query = f"{' '.join(keywords)} 船宿"
        tools.retrieve_trips.func(query=trip_query, **merged_kwargs)

    print(f"检索完成，找到 {len(state_manager.DataStorage.BASKET)} 条资料")


def get_buddy_response_stream(query, context_docs, user_profile):
    """DeepSeek 流式生成"""
    doc_text = ""
    if context_docs:
        doc_text = "\n".join([f"[Ref: {i + 1}] {d.page_content[:300]}" for i, d in enumerate(context_docs)])

    sys_prompt = config.BUDDY_RESPONSE_PROMPT.replace("{user_profile}", str(user_profile))

    input_text = f"【参考资料】：\n{doc_text}\n\n用户问题：{query}"

    for chunk in agent_llm.stream([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=input_text)
    ]):
        if chunk.content:
            yield chunk.content


def extract_new_memory(user_input, ai_response, user_file):
    """静默更新记忆"""
    profile = st.session_state.user_profile
    user_context = f"用户: {user_input}\n教练: {ai_response}"

    try:
        response = agent_llm.invoke([
            SystemMessage(content=config.MEMORY_EXTRACTION_PROMPT),
            HumanMessage(content=user_context)
        ])
        content = response.content.strip()
        if "```json" in content:
            content = re.search(r"\{(.*)\}", content, re.DOTALL).group()

        new_info = json.loads(content)
        mapping = {
            "new_animals": "seen_animals",
            "new_divesites": "visited_sites",
            "new_prefs": "dynamic_notes",
            "new_tips": "dive_tips"
        }

        updated = False
        for j_key, p_key in mapping.items():
            for item in new_info.get(j_key, []):
                if p_key not in profile: profile[p_key] = []
                if item not in profile[p_key]:
                    profile[p_key].append(item)
                    updated = True

        if updated:
            state_manager.save_user_profile(user_file, profile)
            return True
    except:
        pass
    return False