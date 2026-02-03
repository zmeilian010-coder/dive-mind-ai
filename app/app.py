# app.py 顶部
# 只有在云端（Linux）环境才运行补丁
import platform
if platform.system() != "Windows":
    try:
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass

import streamlit as st
import os
import json
from streamlit_mic_recorder import mic_recorder
import ai_engine

# 2. 页面配置
st.set_page_config(page_title="DiveMind AI", page_icon="🤿", layout="centered")

# 3. 导入我们拆分出去的模块
import config
import models
import state_manager
import ai_engine
import ui_components

# ====================== 状态初始化 (放在脚本最顶部) ============================
state_manager.init_session_state()

# 初始化模型和数据库
slm_parser, llm_buddy, rag_db = models.get_models()

# 初始化用户文件路径
user_id = state_manager.get_persistent_user_id()
user_file = os.path.join(config.USER_MEMORY_DIR, f"{user_id}.json")

# 初始化输入状态
if "stt_buffer" not in st.session_state:
    st.session_state.stt_buffer = ""
if "active_prompt" not in st.session_state:
    st.session_state.active_prompt = None

# 增加一个 id 记录器,用于记录语音输入的录音处理状态
if "last_processed_audio_id" not in st.session_state:
    st.session_state.last_processed_audio_id = None

# ==============================侧边栏逻辑=====================================
# 加载侧边栏用户档案
ui_components.render_sidebar(user_file)
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


# ====================== 主界面逻辑 ==============================
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
            choices = ["0-29", "30-49","50-99", "100+"]
            for i, c in enumerate(choices):
                if cols[i].button(c):
                    st.session_state.user_profile['logs'] = c;
                    st.session_state.onboarding_step = 3;
                    st.rerun()

        elif st.session_state.onboarding_step == 3:
            st.write("最后，你最喜欢的**潜水风格**是？")
            choices = ["看大货 (鲨鱼/Manta)", "找微距 (海兔)", "放流潜水", "沉船/洞穴", "水下摄影", "夜潜"]
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

# =======================输入区域 =================================

# 统一入口逻辑
prompt = None

# 如果处于【非复习模式】，才显示输入组件
if not st.session_state.get("review_mode", False):

    # --- 麦克风按钮 (始终显示在输入框上方) ---
    from streamlit_mic_recorder import mic_recorder
    import ai_engine

    # 放置在工具栏列中
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        audio = mic_recorder(
            start_prompt="🎤 语音提问",
            stop_prompt="🛑 停止录音",
            key="global_mic_recorder"
        )

    # 处理录音结果
    if audio:
        # 每一个新的录音，组件都会生成一个唯一的 id (通常在 audio['id'])
        # 如果没有 id，我们就用音频内容的 hash 值作为标识
        import hashlib

        current_audio_id = hashlib.md5(audio['bytes']).hexdigest()

        # 【核心修复】只有当这段音频的 ID 和上一次处理的不一样时，才进入识别
        if current_audio_id != st.session_state.last_processed_audio_id:
            with st.chat_message("assistant"):
                with st.spinner("Buddy 正在转化文字..."):
                    recognized_text = ai_engine.speech_to_text(audio['bytes'])
                    if recognized_text:
                        # 标记这段音频已经处理过了
                        st.session_state.last_processed_audio_id = current_audio_id
                        st.session_state.stt_buffer = recognized_text
                        st.rerun()  # 识别完刷新，进入确认框模式
                    else:
                        st.warning("识别结果为空，请重试。")
                        audio = None
    # --- 语音确认/编辑区域 ---
    if st.session_state.stt_buffer:
        with st.container(border=True):
            st.caption("📝 语音识别结果 (可在此修改):")
            # 注意：这里的 key 必须固定
            confirmed_text = st.text_area("stt_editor",
                                          value=st.session_state.stt_buffer,
                                          label_visibility="collapsed")

            c1, c2 = st.columns(2)
            if c1.button("✅ 确认并发送", key="stt_send"):
                st.session_state.active_prompt = confirmed_text
                st.session_state.stt_buffer = ""
                st.rerun()
            if c2.button("🗑️ 取消", key="stt_cancel"):
                st.session_state.stt_buffer = ""
                st.rerun()

    # --- 唯一的文本输入框渲染 ---
    # 关键逻辑：如果正在“确认语音”，则不渲染底部的输入框，防止 ID 冲突
    if not st.session_state.stt_buffer:
        user_input = st.chat_input("和你的 Buddy 聊聊...", key="main_chat_input")
        if user_input:
            prompt = user_input

# 捕获刚刚确认的语音输入
if st.session_state.active_prompt:
    prompt = st.session_state.active_prompt
    st.session_state.active_prompt = None  # 阅后即焚

# --- 正常对话逻辑 ---
if prompt :
    # 1. 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Buddy 正在思考...", expanded=True) as status:
            # --- 第一步：小模型解析意图 ---
            status.write("🔍 正在分析你的意图...")
            analysis = ai_engine.parse_user_intent(prompt, st.session_state.messages,state_manager.load_user_profile(user_file))

            # --- 第二步：系统自动执行检索 ---
            if analysis["intent"] == "CONSULT":
                status.write(f"📚 正在为你翻阅知识库: {analysis['keywords']}...")
                ai_engine.automated_retrieval_hub(analysis, st.session_state.user_profile)
            else:
                status.write("💬 原来是想找我叙叙旧，这就来！")

            status.update(label="思考完成!正在打字......", state="complete", expanded=False)

        # --- 第三步：大模型生成回复 ---
        import state_manager

        user_profile = st.session_state.user_profile
        response = ai_engine.get_buddy_response_stream(prompt, state_manager.DataStorage.BASKET, user_profile)

        full_response = st.write_stream(response)
        final_answer = full_response
        st.markdown(final_answer)

        # --- 渲染超级卡片 ---
        if state_manager.DataStorage.BASKET:
            ui_components.display_trip_results(state_manager.DataStorage.BASKET)

        # 存入历史记录
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        # --- 第五步：静默记忆提取 (异步) ---
        ai_engine.extract_new_memory(prompt, final_answer,st.session_state.user_profile)