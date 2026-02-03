# ai_engine.py
import json
import re
import datetime
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
import io
from openai import OpenAI
import config
import models
import state_manager
import tools

# 获取模型实例
slm_parser, agent_llm, _ = models.get_models()


def speech_to_text(audio_bytes):
    """
    将录音字节流发送至硅基流动进行识别
    """
    if not audio_bytes:
        print("❌ 错误：收到空的音频数据")
        return None

    print(f"🎤 收到音频数据，大小: {len(audio_bytes)} 字节")

    try:
        # 1. 准备客户端 (这里直接用 OpenAI SDK，因为它兼容硅基流动)
        client = OpenAI(
            api_key=st.secrets["SILICONFLOW_API_KEY"],
            base_url=config.SILICON_BASE_URL
        )

        # 2. 将字节流转换为类似文件的对象
        # 注意：streamlit-mic-recorder 默认通常返回 webm 或 wav 格式
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"

        # 3. 调用识别接口
        # 我们在这里注入“潜水词库”作为 Prompt 诱导，极大提升准确率
        transcript = client.audio.transcriptions.create(
            model=config.STT_MODEL,
            file=audio_file,
            prompt="用户的输入可能包含潜水专业词汇比如BCD、湿衣等"
        )

        print(f"✅ 识别成功！结果: {transcript.text}")
        return transcript.text
    except Exception as e:
        st.error(f"语音识别失败: {e}")
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