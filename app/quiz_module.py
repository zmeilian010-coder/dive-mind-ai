import streamlit as st


def render_quiz_page():
    """复习模式的主界面函数"""
    st.subheader("🧠 Buddy 知识复习场")
    st.info("模式已切换：当前处于【行前知识复习】状态。")

    # 模拟一个出题卡片
    with st.container(border=True):
        st.write("⚓ **这里是未来的出题区**")
        st.caption("稍后我们会在这里接入 7B 模型生成的情境题和选项。")

        # 放置一个临时的“换一题”按钮，验证交互
        if st.button("⏭️ 随便换一题试试"):
            st.toast("逻辑通畅，等待接入数据库...")