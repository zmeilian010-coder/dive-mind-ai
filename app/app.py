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

# 2. 页面配置
st.set_page_config(page_title="DiveMind AI", page_icon="🤿", layout="centered")

# 3. 导入我们拆分出去的模块
import config
import models
import state_manager
import ai_engine
import ui_components

# --- 【在逻辑开始前初始化所有状态】 ---
state_manager.init_session_state()

# 4. 初始化模型和数据库
slm_parser, llm_buddy, rag_db = models.get_models()

# 5. 初始化用户文件路径
user_id = state_manager.get_persistent_user_id()
user_file = os.path.join(config.USER_MEMORY_DIR, f"{user_id}.json")



# -----主界面逻辑---
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