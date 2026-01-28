import streamlit as st
import random
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
import models
import state_manager
import config

# 获取模型和数据库实例
slm_parser, agent_llm, rag_db = models.get_models()


# --- 1. 抽题引擎：打捞 10 个原始知识块 ---
def get_quiz_pool():
    """根据优先级打捞 10 个知识块"""
    profile = st.session_state.user_profile
    incidents = profile.get("incident_history", [])

    final_pool = []

    # 【优先级 1】针对意外记录进行打捞
    if incidents:
        # 取最近的 2 条意外作为关键词搜索
        search_query = " ".join(incidents[-2:])
        # 搜索 MD 格式的知识库
        incident_docs = rag_db.similarity_search(
            search_query,
            k=3,
            filter={"Metadata_file_type": "md"}
        )
        final_pool.extend(incident_docs)
        print(f"🎯 优先级 1：基于意外记录打捞了 {len(incident_docs)} 个知识块")

    # 【优先级 2】打捞带习题标签的块 (is_quiz=True)
    quiz_docs = rag_db.get(
        where={"is_quiz": True},
        limit=20
    )
    if quiz_docs['documents']:
        # 包装成 Document 对象并随机选 5 个
        all_quizzes = [models.Document(page_content=d, metadata=m)
                       for d, m in zip(quiz_docs['documents'], quiz_docs['metadatas'])]
        random.shuffle(all_quizzes)
        final_pool.extend(all_quizzes[:5])
        print(f"📚 优先级 2：打捞了 {len(all_quizzes[:5])} 个预设习题块")

    # 【优先级 3】随机补位 (从通用 MD 库抽)
    if len(final_pool) < 10:
        needed = 10 - len(final_pool)
        # 排除掉不需要出题的分类（黑名单逻辑）
        random_docs = rag_db.similarity_search(
            "潜水基础理论与安全规范",
            k=20,
            filter={"Metadata_file_type": "md"}
        )
        # 简单去重并补足
        for doc in random_docs:
            if doc.page_content[:50] not in [d.page_content[:50] for d in final_pool]:
                final_pool.append(doc)
            if len(final_pool) >= 10: break

    random.shuffle(final_pool)  # 打乱顺序
    return final_pool[:10]


# --- 2. 出题官：让 7B 把 Chunk 变成选择题 ---
def generate_question_json(doc):
    """调用 7B 模型生成或提取题目"""
    is_premade_quiz = doc.metadata.get("is_quiz", False)

    if is_premade_quiz:
        prompt = f"""你是一个题目解析员。
        这段文字里包含 1 道或多道潜水习题。请【随机挑选其中 1 道】，并严格按 JSON 格式提取：
        文字内容：{doc.page_content}
        要求：
        - question: 题干
        - options: [A.xx, B.xx, C.xx, D.xx] 形式的列表
        - answer: 仅返回正确选项的字母（如 "B"）
        - explanation: 简短的解析
        """
    else:
        prompt = f"""你是一个专业的潜水教练 Buddy。
        请基于以下知识点，编一个具体的【潜水场景问题】。
        知识点：{doc.page_content}
        要求：
        - 出一道单选题。
        - 必须是场景化的（例如：你在水下遇到了XX情况...）。
        - 干扰项要具有迷惑性，但不能模棱两可。
        - 必须返回 JSON：{{"question": "", "options": ["A.xx", "B.xx", "C.xx", "D.xx"], "answer": "字母", "explanation": ""}}
        """

    try:
        response = slm_parser.invoke([SystemMessage(content=prompt)])
        # 清洗并解析 JSON (复用之前的正则逻辑)
        content = response.content.strip()
        json_str = re.search(r'\{.*\}', content, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return None


def render_quiz_page():
    st.title("随时复习")

    # 1. 初始化复习进度
    if "quiz_pool" not in st.session_state or not st.session_state.quiz_pool:
        with st.spinner("Buddy 正在根据你的档案和意外记录为你备课..."):
            st.session_state.quiz_pool = get_quiz_pool()
            st.session_state.quiz_step = 0
            st.session_state.score_box = {"correct": 0, "wrong": 0}
            st.session_state.wrong_log = []
            st.session_state.quiz_stage = "asking"
        st.rerun()

    # 2. 如果 10 题做完了，显示总结报告
    if st.session_state.quiz_step >= 10:
        render_summary()
        return

    # 3. 获取当前题目
    current_idx = st.session_state.quiz_step
    # 缓存题目数据，防止重复调用 AI
    if "current_q_data" not in st.session_state:
        chunk = st.session_state.quiz_pool[current_idx]
        q_data = generate_question_json(chunk)
        if not q_data:  # 如果生成失败，跳过这一题
            st.session_state.quiz_step += 1
            st.rerun()
        # 存入题目数据，并带上原文用于复盘
        q_data["raw_content"] = chunk.page_content
        q_data["images"] = chunk.metadata.get("images")
        st.session_state.current_q_data = q_data

    q = st.session_state.current_q_data

    # --- 开始渲染界面 ---
    st.progress(st.session_state.quiz_step / 10, text=f"进度: 第 {st.session_state.quiz_step + 1} / 10 题")

    with st.container(border=True):
        st.markdown(f"#### Q{st.session_state.quiz_step + 1}: {q['question']}")
        if q.get("images"):
            # 如果有图，显示第一张
            st.image(q["images"].split(",")[0], caption="参考图示")

        # 答题阶段
        if st.session_state.quiz_stage == "asking":
            # 动态生成 4 个按钮
            for opt in q['options']:
                if st.button(opt, use_container_width=True):
                    # 判断对错
                    user_choice = opt[0]  # 取 A, B, C, D
                    if user_choice == q['answer']:
                        st.session_state.score_box["correct"] += 1
                        st.session_state.last_result = "✅ 太棒了，完全正确！"
                    else:
                        st.session_state.score_box["wrong"] += 1
                        st.session_state.last_result = f"❌ 哎呀，这题选 {q['answer']} 哦。"
                        # 记录错题
                        st.session_state.wrong_log.append(f"题目: {q['question']} | 正确答案: {q['answer']}")

                    st.session_state.quiz_stage = "feedback"
                    st.rerun()

            if st.button("⏭️ 换一题"):
                del st.session_state.current_q_data
                st.session_state.quiz_step += 1
                st.rerun()

        # 反馈阶段
        else:
            if "✅" in st.session_state.last_result:
                st.success(st.session_state.last_result)
            else:
                st.error(st.session_state.last_result)

            st.markdown(f"**Buddy 的解析：** {q['explanation']}")

            with st.expander("📖 查看手册原文复盘"):
                st.info(q['raw_content'])

            if st.button("➡️ 继续下一题", type="primary"):
                del st.session_state.current_q_data
                st.session_state.quiz_step += 1
                st.session_state.quiz_stage = "asking"
                st.rerun()


def render_summary():
    """渲染 10 题结束后的总结报告"""
    st.balloons()
    st.header("📊 本轮复习总结")
    c1, c2 = st.columns(2)
    c1.metric("答对", st.session_state.score_box["correct"])
    c2.metric("答错", st.session_state.score_box["wrong"])

    if st.session_state.wrong_log:
        st.subheader("💡 需要加强的知识点：")
        # 这里以后可以调用 DeepSeek 汇总总结，现在先简单列出
        for log in st.session_state.wrong_log:
            st.caption(f"• {log}")

    if st.button("🔄 再来一轮", type="primary"):
        # 清空状态，重新开始
        for key in ["quiz_pool", "quiz_step", "score_box", "wrong_log", "current_q_data"]:
            if key in st.session_state: del st.session_state[key]
        st.rerun()