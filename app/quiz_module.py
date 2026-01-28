# quiz_model.py
import streamlit as st
import random
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
import models
import state_manager
import config
from langchain_core.documents import Document
# 获取模型和数据库实例
slm_parser, agent_llm, rag_db = models.get_models()


# --- 1. 抽题引擎：检索 10 个原始知识块 ---
def get_quiz_pool():
    """根据优先级检索 10 个知识块"""
    profile = st.session_state.user_profile
    incidents = profile.get("incident_history", [])

    final_pool = []

    # 【优先级 1】针对意外记录进行检索
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
        print(f"🎯 优先级 1：基于意外记录检索了 {len(incident_docs)} 个知识块")

    # 【优先级 2】检索带习题标签的块 (is_quiz=True)
    quiz_docs = rag_db.get(
        where={"is_quiz": True},
        limit=20
    )
    if quiz_docs['documents']:
        # 包装成 Document 对象并随机选 5 个
        all_quizzes = [Document(page_content=d, metadata=m)
                       for d, m in zip(quiz_docs['documents'], quiz_docs['metadatas'])]
        random.shuffle(all_quizzes)
        final_pool.extend(all_quizzes[:5])
        print(f"📚 优先级 2：检索了 {len(all_quizzes[:5])} 个预设习题块")

    # 【优先级 3】随机补位 (从通用 MD 库抽)
    if len(final_pool) < 10:
        print("因召回不足10个，需要随机补位")
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
        - answer: 仅返回正确选项的字母（如 "B"或者"B,C"）
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
        - 必须返回 JSON：{{"question": "", "options": ["A.xx", "B.xx", "C.xx", "D.xx"], "answer": 仅返回正确选项的字母（如 "B"或者"B,C"）, "explanation": ""}}
        """

    try:
        response = slm_parser.invoke([SystemMessage(content=prompt)])
        # 清洗并解析 JSON (复用之前的正则逻辑)
        content = response.content.strip()
        json_str = re.search(r'\{.*\}', content, re.DOTALL).group()
        data = json.loads(json_str)
        data["raw_content"] = doc.page_content
        return data
    except:
        return None


# --- 3. 预加载所有题目 (解决刷新慢) ---
def preload_questions(raw_pool):
    processed_questions = []
    progress_bar = st.progress(0, text="Buddy 正在为你备课...")

    for i, doc in enumerate(raw_pool):
        q = generate_question_json(doc)
        if q:
            processed_questions.append(q)
        progress_bar.progress((i + 1) / len(raw_pool), text=f"正在生成第 {i + 1}/10 题...")

    progress_bar.empty()
    return processed_questions

# --- 4. 主渲染函数 ---
def render_quiz_page():
    if "quiz_pool" not in st.session_state or not st.session_state.quiz_pool:
        raw_pool = get_quiz_pool()
        st.session_state.quiz_pool = preload_questions(raw_pool)
        st.session_state.quiz_step = 0
        st.session_state.score_box = {"correct": 0, "wrong": 0}
        st.session_state.wrong_log = []
        st.session_state.quiz_stage = "asking"
        st.rerun()

    if st.session_state.quiz_step >= len(st.session_state.quiz_pool):
        render_summary()
        return

    q = st.session_state.quiz_pool[st.session_state.quiz_step]

    # --- 界面展示 ---
    st.title("🧠 Buddy 知识练兵场")
    st.progress(st.session_state.quiz_step / len(st.session_state.quiz_pool))

    with st.container(border=True):
        st.markdown(f"**第 {st.session_state.quiz_step + 1} 题**")
        st.subheader(q['question'])

        # 答题阶段
        q_type = q.get('type', 'single')
        if st.session_state.quiz_stage == "asking":
            if q_type == "multiple":
                st.info("这是一道多选题，请勾选所有正确答案后点击提交。")
                selected = []
                for opt in q['options']:
                    if st.checkbox(opt, key=f"q_{st.session_state.quiz_step}_{opt}"):
                        selected.append(opt[0])  # 抓取 A, B, C

                if st.button("提交回答", type="primary"):
                    st.session_state.user_answer = sorted(selected)
                    st.session_state.is_correct = (st.session_state.user_answer == sorted(q['answer']))
                    if st.session_state.is_correct:
                        st.session_state.score_box["correct"] += 1
                    else:
                        st.session_state.score_box["wrong"] += 1
                        st.session_state.wrong_log.append(q['question'])
                    st.session_state.quiz_stage = "feedback"
                    st.rerun()
            else:
                # 单选题使用按钮，点击即确认
                for opt in q['options']:
                    if st.button(opt, use_container_width=True, key=f"btn_{st.session_state.quiz_step}_{opt}"):
                        st.session_state.user_answer = [opt[0]]
                        st.session_state.is_correct = (opt[0] in q['answer'])
                        if st.session_state.is_correct:
                            st.session_state.score_box["correct"] += 1
                        else:
                            st.session_state.score_box["wrong"] += 1
                            st.session_state.wrong_log.append(q['question'])
                        st.session_state.quiz_stage = "feedback"
                        st.rerun()

        # 反馈阶段
        else:
            if st.session_state.is_correct:
                st.success("✅ 完全正确！Buddy 为你点赞！")
            else:
                st.error(f"❌ 选错啦。正确答案是：{', '.join(q['answer'])}")
                st.write(
                    f"你的回答：{', '.join(st.session_state.user_answer) if st.session_state.user_answer else '未选择'}")

            st.markdown(f"💡 **解析：** {q['explanation']}")

            # 解决问题 1：如果是预设习题，不显示原文，避免泄露后续题目
            if not q.get("is_premade"):
                with st.expander("📖 查看手册原文复盘"):
                    st.info(q['raw_content'])
            else:
                st.caption("注：此题来自教材原题库，建议查阅相关章节手册深度复习。")

            if st.button("下一题 ➡️", use_container_width=True):
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