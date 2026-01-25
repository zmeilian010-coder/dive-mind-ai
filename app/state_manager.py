# state_manager.py 完整补充版
import os
import uuid
import json
import streamlit as st
import config


class DataStorage:
    BASKET = []


def get_persistent_user_id():
    """获取设备唯一指纹"""
    node_id = str(uuid.getnode())
    return f"diver_{node_id[:8]}"


def load_user_profile(user_file):
    """加载 JSON 档案"""
    if os.path.exists(user_file):
        with open(user_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_user_profile(user_file, profile):
    """保存 JSON 档案"""
    with open(user_file, 'w', encoding='utf-8') as f:
        json.dump(profile, f, ensure_ascii=False, indent=4)


def init_session_state():
    """
    状态大管家：确保所有变量在程序运行前都已定义
    """
    # 1. 识别并锁定用户 ID
    if "user_id" not in st.session_state:
        st.session_state.user_id = get_persistent_user_id()

    user_file = os.path.join(config.USER_MEMORY_DIR, f"{st.session_state.user_id}.json")
    st.session_state.user_file = user_file  # 存入 state 方便以后调用

    # 2. 初始化 Onboarding 状态 (核心修复点)
    if "onboarding_complete" not in st.session_state:
        profile = load_user_profile(user_file)
        if profile:
            st.session_state.user_profile = profile
            st.session_state.onboarding_complete = True
            print(f">>> 成功识别老用户: {st.session_state.user_id}")
        else:
            # 新用户默认档案
            print(f">>> 识别为新用户: {st.session_state.user_id}，正在创建档案")
            st.session_state.user_profile = {
                "level": None,
                "logs": None,
                "preference": [],
                "seen_animals": [],
                "visited_sites": [],
                "dynamic_notes": [],
                "dive_tips": [],
                "incident_history": []
            }
            st.session_state.onboarding_complete = False
            st.session_state.onboarding_step = 1

    # 3. 初始化对话和模式状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "review_mode" not in st.session_state:
        st.session_state.review_mode = False
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "pending_ai_reply" not in st.session_state:
        st.session_state.pending_ai_reply = False