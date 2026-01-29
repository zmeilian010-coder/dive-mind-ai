import streamlit as st
import json
import random
import os
import re
import config
import models
import state_manager
from langchain_core.documents import Document

# 获取数据库实例
_, _, rag_db = models.get_models()


def load_quiz_bank():
    """从本地 JSON 加载题库"""
    quiz_path = os.path.join(config.DATA_DIR, "processed", "quiz_bank.json")
    if os.path.exists(quiz_path):
        with open(quiz_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def prepare_quiz_set():
    """
    备课引擎：一次性准备好 10 道题及其背景资料
    """
    all_questions = load_quiz_bank()
    if not all_questions:
        return []

    profile = st.session_state.user_profile
    incidents = profile.get("incident_history", [])

    selected_questions = []

    # --- 优先级 1：针对性匹配 (P1) ---
    if incidents:
        # 将所有意外描述拼成一个大字符串，提取关键词（简单去噪）
        incident_text = " ".join(incidents).lower()
        # 找出包含这些关键词的题目
        p1_candidates = []
        for q in all_questions:
            # 检查题干或所属标题是否命中了意外关键词
            # 比如意外里有“耳压”，题目里有“耳压”
            q_text = (q['question'] + str(q['metadata'])).lower()
            if any(kw.lower() in q_text for kw in ["耳压", "减压", "气耗", "潜伴"] if kw in incident_text):
                p1_candidates.append(q)

        # 随机选最多 5 道 P1 题目
        random.shuffle(p1_candidates)
        selected_questions.extend(p1_candidates[:5])
        print(f"🎯 P1 针对性备课：选中 {len(selected_questions)} 道关联题目")

    # --- 优先级 2：系统性补位 (P2) ---
    # 排除掉已经选中的 ID
    selected_ids = {q['question_id'] for q in selected_questions}
    remaining_pool = [q for q in all_questions if q['question_id'] not in selected_ids]

    needed = 10 - len(selected_questions)
    if needed > 0:
        p2_selection = random.sample(remaining_pool, min(needed, len(remaining_pool)))
        selected_questions.extend(p2_selection)
        print(f"📚 P2 随机补位：选中 {len(p2_selection)} 道题")

    # --- 核心：0 延迟资料预装载 (Back-tracing) ---
    # 这一步保证了答题时不需要再查库
    final_quiz_set = []
    for q in selected_questions:
        # 尝试从数据库获取原理原文和图片
        try:
            # 使用提取题目时存下的 parent_chunk_id
            parent_id = q.get("parent_chunk_id")
            parent_data = rag_db.get(ids=[parent_id], include=['documents', 'metadatas'])

            if parent_data['documents']:
                # 找到该习题在 MD 里的“邻居”：同一个 Header 下的非习题块
                # 这里我们先简单拿父块内容，后期可以优化为拿父级标题下的知识块
                q["knowledge_context"] = parent_data['documents'][0]
                q["images"] = parent_data['metadatas'][0].get("images")
        except Exception as e:
            print(f"⚠️ 预装载资料失败 (ID: {q['question_id']}): {e}")
            q["knowledge_context"] = "暂无手册原文参考"
            q["images"] = None

        final_quiz_set.append(q)

    random.shuffle(final_quiz_set)  # 打乱展示顺序
    return final_quiz_set


def fetch_principle_context(q_metadata, question_text):
    from models import get_models
    _, _, rag_db = get_models()

    # 1. 拿到最深层的非习题标题
    search_headers = [q_metadata.get(f"Header{i}") for i in range(1, 5)
                      if q_metadata.get(f"Header{i}") and "习题" not in q_metadata.get(f"Header{i}")]

    if not search_headers: return None
    anchor_header = search_headers[-1]

    # 2. 组合搜索：标题 + 题干的前10个字（代表核心知识点）
    search_query = f"{anchor_header} {question_text[:15]}"

    # 3. 严格限制在当前章节内搜索非习题内容
    results = rag_db.similarity_search(
        search_query,
        k=1,
        filter={
            "$and": [
                {"is_quiz": {"$eq": False}},
                {f"Header{len(search_headers)}": {"$eq": anchor_header}}
            ]
        }
    )
    return results[0].page_content if results else None

def render_quiz_page():
    """复习模式主渲染函数"""
    # 1. 检查是否需要备课
    if "current_quiz_set" not in st.session_state or not st.session_state.current_quiz_set:
        with st.status("🤿 Buddy 正在为你备课...", expanded=True) as status:
            quiz_set = prepare_quiz_set()
            if not quiz_set:
                st.error("题库空空如也，快去添加资料吧！")
                return

            st.session_state.current_quiz_set = quiz_set
            st.session_state.quiz_step = 0
            st.session_state.score_box = {"correct": 0, "wrong": 0}
            st.session_state.wrong_log = []
            st.session_state.quiz_stage = "asking"  # 状态：asking 或 feedback
            status.update(label="备考完成！开启挑战！", state="complete")
        st.rerun()

    # 2. 总结界面
    if st.session_state.quiz_step >= len(st.session_state.current_quiz_set):
        render_summary()
        return

    # 3. 答题界面展示
    curr_q = st.session_state.current_quiz_set[st.session_state.quiz_step]

    st.write(f"**第 {st.session_state.quiz_step + 1} / 10 题**")
    st.progress((st.session_state.quiz_step) / 10)

    with st.container(border=True):
        st.subheader(curr_q['question'])

        # 展示预装载的图片
        if curr_q.get("images"):
            st.image(curr_q["images"].split(",")[0], caption="参考图示")

        # 答题逻辑
        if st.session_state.quiz_stage == "asking":
            render_options(curr_q)
        else:
            render_feedback(curr_q)


def render_options(q):
    """渲染选项按钮"""
    q_type = q.get('type', 'single')

    if q_type == 'multiple':
        st.info("💡 这是一道多选题哦！")
        selected = []
        for opt in q['options']:
            if st.checkbox(opt, key=f"check_{opt}"):
                # 判断：如果是 "a.xxx" 格式取首字母，否则取全文
                val = opt[0].lower() if (len(opt) > 1 and opt[1] == '.') else opt.strip()
                selected.append(val)

        if st.button("提交答案", type="primary"):
            check_answer(selected, q['answer'])
    else:
        # 单选直接用按钮
        for opt in q['options']:
            if st.button(opt, use_container_width=True, key=f"btn_{opt}"):
                # 判断：如果是 "a.xxx" 格式取首字母，否则取全文
                val = opt[0].lower() if (len(opt) > 1 and opt[1] == '.') else opt.strip()
                check_answer([val], q['answer'])


def check_answer(user_choice_raw, correct_ans_list):
    """
    问题 3 修复：标准化对比逻辑，解决判断题失效
    """
    # --- 探针逻辑开始 ---
    user_raw = str(user_choice_raw)
    user_clean = user_raw.strip().lower()

    # 尝试提取字母标签
    import re
    label_match = re.match(r'^([a-d])[\.\s]', user_clean)
    user_label = label_match.group(1) if label_match else user_clean

    # 准备标准答案
    clean_correct = [str(a).strip().lower() for a in correct_ans_list]

    # 判定结果
    is_ok = (user_label in clean_correct) or (user_clean in clean_correct)

    # 打印探针数据（黑窗口）
    print(f"\n🔍 [判分探针数据]")
    print(f"   - 用户原始点击: {repr(user_raw)}")
    print(f"   - 提取到的标签: {repr(user_label)}")
    print(f"   - 数据库正确答案: {clean_correct}")
    print(f"   - 最终判定: {'✅对' if is_ok else '❌错'}")

    # 存入 session_state 供前端反馈阶段展示
    st.session_state.debug_probe = {
        "user_label": user_label,
        "correct_ans": clean_correct
    }
    # --- 探针逻辑结束 ---

    """
    user_choices: 可能是单选字符串 "A.xx"，也可能是多选列表 ["a", "c"]
    correct_ans_list: 数据库里的标准答案列表 ["a", "c"] 或 ["正确"]
    """
    # 1. 统一格式：确保用户输入是列表
    if isinstance(user_choice_raw, str):
        raw_list = [user_choice_raw]
    else:
        raw_list = user_choice_raw

    # 2. 提取并清理用户答案
    processed_user_ans = []
    for item in raw_list:
        clean_item = str(item).strip().lower()

        # --- 【智能提取补丁】 ---
        # 只有当它是 A. B. C. D. 开头时，才提取首字母
        if re.match(r'^[a-d][\.\s]', clean_item):
            val = clean_item[0]
        else:
            # 如果是“正确/错误”，保留全文
            val = clean_item
        # -----------------------

        processed_user_ans.append(val)

    # 4. 判定 (比较两个列表的内容是否一致)
    is_correct = set(processed_user_ans) == set(clean_correct)

    # --- 探针输出 ---
    print(f"DEBUG: 用户处理后答案: {processed_user_ans} | 标准答案: {clean_correct} | 结果: {is_ok}")

    if is_correct:
        st.session_state.score_box["correct"] += 1
        st.session_state.last_feedback = "✅ 完全正确！"
    else:
        st.session_state.score_box["wrong"] += 1
        st.session_state.last_feedback = f"❌ 选错啦，正确答案是：{', '.join(correct_ans_list).upper()}"
        # 记录错题信息用于最后总结 (问题 4 准备数据)
        curr_q = st.session_state.current_quiz_set[st.session_state.quiz_step]
        st.session_state.wrong_log.append({
            "question": curr_q['question'],
            "category": curr_q['metadata'].get('category', '通用')
        })

    st.session_state.quiz_stage = "feedback"
    st.rerun()


def render_feedback(q):
    """
    问题 1 & 移动解析：反馈界面展示
    """
    if "✅" in st.session_state.last_feedback:
        st.success(st.session_state.last_feedback)
    else:
        st.error(st.session_state.last_feedback)

    # --- 【新增：选项对照区】 ---
    st.write("**选项回顾：**")
    for opt in q['options']:
        opt_label_match = re.match(r'^([a-d])[\.\s]', opt.lower())
        opt_label = opt_label_match.group(1) if opt_label_match else opt.lower()

        # 如果这个选项是正确答案之一，给它加个绿色的框或加粗
        if opt_label in [a.lower() for a in q['answer']]:
            st.markdown(f":green[👉 **{opt}** (正确答案)]")
        else:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{opt}")

    # --- 这里的解析现在在答案揭晓后显示了 ---
    if q.get('explanation'):
        st.info(f"💡 **Buddy 的解析：** {q['explanation']}")

    # # 问题 1 修复：不展示习题块原文，展示回溯到的原理原文
    # with st.expander("📖 查看手册原理复盘"):
    #     # 尝试回溯原理
    #     principle = fetch_principle_context(q['metadata'])
    #     if principle:
    #         st.markdown(principle)
    #     else:
    #         st.caption("Buddy 正在利用专业知识为你解析...")
    #         # 如果库里没原理，可以让 AI 现写一段
    #         st.write(q.get('explanation', "暂无原文参考，请查阅 PADI 手册。"))

    if "debug_probe" in st.session_state:
        st.caption(
            f"🧪 调试信息 -> 你选了: `{st.session_state.debug_probe['user_label']}` | 正确答案: `{st.session_state.debug_probe['correct_ans']}`")

    if st.button("下一题 ➡️", use_container_width=True):
        st.session_state.quiz_step += 1
        st.session_state.quiz_stage = "asking"
        st.rerun()


def render_summary():
    """
    问题 4 修复：使用 AI 生成自然语言总结报告
    """
    st.balloons()
    st.header("📊 复习完成报告")

    c1, c2 = st.columns(2)
    c1.metric("答对", st.session_state.score_box["correct"])
    c2.metric("答错", st.session_state.score_box["wrong"])

    # --- AI 总结逻辑 ---
    if st.session_state.wrong_log:
        st.subheader("🕵️ Buddy 的深度总结")
        # 提取错题的分类
        wrong_categories = [item['category'] for item in st.session_state.wrong_log]
        wrong_questions = [q['question'][:30] for q in st.session_state.wrong_log]

        # 构造给 DeepSeek 的提示词
        from models import get_models
        _, agent_llm, _ = get_models()

        summary_prompt = config.QUIZ_SUMMARY_TEMPLATE.format(
            correct_count=st.session_state.score_box["correct"],
            wrong_count=st.session_state.score_box["wrong"],
            wrong_categories=list(set(wrong_categories)), # 去重
            wrong_questions=wrong_questions
        )

        with st.spinner("Buddy 正在分析你的复习表现..."):
            summary_res = agent_llm.invoke(summary_prompt)
            st.write(summary_res.content)
    else:
        st.success("完美！你对本次复习的知识点掌握得天衣无缝，Buddy 为你感到自豪！")

    if st.button("🔄 开启新一轮挑战", type="primary"):
        st.session_state.current_quiz_set = []  # 触发重新备课
        st.rerun()