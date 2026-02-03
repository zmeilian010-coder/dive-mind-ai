import streamlit as st
import json
import random
import os
import re
import config
import models
import jieba
import state_manager
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

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

    final_quiz_set = []
    for q in selected_questions:
        final_quiz_set.append(q)

    return final_quiz_set

    random.shuffle(final_quiz_set)  # 打乱展示顺序
    return final_quiz_set


def get_clean_category(metadata):
    """
    逻辑：从 H4 往 H1 找，第一个不含“习题”二字的标题就是最精准的分类
    """
    # 按优先级从深到浅排列
    header_keys = ["Header4", "Header3", "Header2", "Header1"]

    for key in header_keys:
        h_val = metadata.get(key)
        if h_val and "习题" not in str(h_val):
            return h_val

    # 如果都没找到，再拿文件名兜底
    return metadata.get("category", "通用知识")

def fetch_principle_context(q_metadata, q_text):
    """
    【3.1 全标题加权精排版】
    在当前 Header1 范围内，检索 H2-H4 标题命中或正文命中的最相关原文。
    排除 H1 和包含“习题”的标题干扰。
    """
    from models import get_models
    _, _, rag_db = get_models()

    h1 = q_metadata.get("Header1")
    if not h1:
        return None, None

    # 1. 召回：获取当前章节所有【非习题】的知识块
    chapter_data = rag_db.get(
        where={"$and": [
            {"is_quiz": {"$eq": False}},
            {"Header1": {"$eq": h1}}
        ]},
        include=['documents', 'metadatas']
    )

    if not chapter_data['documents']:
        return None, None

    # 2. 提取题干核心关键词
    keywords = [w for w in jieba.lcut(q_text) if len(w) > 1 and w not in ["为什么", "是什么", "如何", "应该", "可以"]]
    print(f"🔍 原理回溯目标关键词: {keywords}")

    scored_docs = []

    for doc_text, meta in zip(chapter_data['documents'], chapter_data['metadatas']):
        total_score = 0

        # --- A. 基础分 (10分保底) ---
        total_score += 10

        # --- B. 构造“有效标题池” (针对要求 1) ---
        # 提取 H2, H3, H4，并过滤掉包含“习题与答案”的内容
        valid_headers_text = ""
        for h_key in ["Header2", "Header3", "Header4"]:
            h_val = str(meta.get(h_key, ""))
            if h_val and "习题与答案" not in h_val:
                valid_headers_text += f" {h_val}"

        valid_headers_text = valid_headers_text.lower()
        page_content = doc_text.lower()

        match_details = []
        for kw in keywords:
            kw_l = kw.lower()

            # 情况 1: 命中 H2/H3/H4 标题 (排除习题标题) -> 给极高权重
            if kw_l in valid_headers_text:
                total_score += 150
                match_details.append(f"🎯标题命中:{kw}")

            # 情况 2: 命中正文内容 -> 给中等权重
            elif kw_l in page_content:
                total_score += 50
                match_details.append(f"📝正文命中:{kw}")

        if total_score > 10:
            scored_docs.append({
                "content": doc_text,
                "images": meta.get("images"),
                "score": total_score,
                "details": match_details
            })

    # 4. 排序并返回最高分者
    if not scored_docs:
        return None, None

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    winner = scored_docs[0]

    print(f"🏆 原理回溯获胜者得分: {winner['score']} | 命中详情: {winner['details']}")

    return winner["content"], winner["images"]

def render_quiz_page():
    """复习模式主渲染函数"""

    if config.DEBUG_MODE:
        with st.sidebar:
            if st.button("⚡ 调试：模拟答题并看总结"):
                # 1. 确保题库已经加载，如果没有，先加载一次
                if "current_quiz_set" not in st.session_state or not st.session_state.current_quiz_set:
                    st.session_state.current_quiz_set = prepare_quiz_set()

                # 2. 模拟分数：比如 7 对 3 错
                st.session_state.score_box = {"correct": 7, "wrong": 3}

                # 3. 模拟错题记录：从当前生成的题库里随手抓 3 道题当作错题
                mock_wrongs = []
                # 假设前 3 题答错了
                for i in range(3):
                    q = st.session_state.current_quiz_set[i]

                    # 构造一个完整的错题对象（和你 check_answer 里的格式一样）
                    mock_wrongs.append({
                        "question": q.get('question'),
                        "options": q.get('options'),
                        "correct_answer": q.get('answer'),
                        "user_answer": ["X"],  # 模拟一个错误选项
                        "explanation": q.get('explanation', '这是调试生成的假解析'),
                        "category": get_clean_category(q.get('metadata', {}))  # 调用你刚才写的分类回溯函数
                    })

                st.session_state.wrong_log = mock_wrongs

                # 4. 强制跳转到总结阶段
                st.session_state.quiz_step = 10
                st.session_state.quiz_stage = "summary"
                st.rerun()

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

    # 打印探针数据（黑窗口）
    print(f"\n🔍 [判分探针数据]")
    print(f"   - 用户原始点击: {repr(user_raw)}")
    print(f"   - 提取到的标签: {repr(user_label)}")
    print(f"   - 数据库正确答案: {clean_correct}")
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
    print(f"DEBUG: 用户处理后答案: {processed_user_ans} | 标准答案: {clean_correct} | 结果: {is_correct}")

    if is_correct:
        st.session_state.score_box["correct"] += 1
        st.session_state.last_feedback = "✅ 完全正确！"
    else:
        st.session_state.score_box["wrong"] += 1
        st.session_state.last_feedback = f"❌ 选错啦，正确答案是：{', '.join(correct_ans_list).upper()}"
        # 记录错题信息用于最后总结 (问题 4 准备数据)
        # 获取当前完整的题目对象
        curr_q = st.session_state.current_quiz_set[st.session_state.quiz_step]

        refined_cat = get_clean_category(curr_q.get('metadata', {}))
        # 【修改点】存入整个题目对象，方便总结页调用
        st.session_state.wrong_log.append({
            "question": curr_q['question'],
            "options": curr_q['options'],
            "correct_answer": correct_ans_list,
            "user_answer": processed_user_ans,
            "explanation": curr_q.get('explanation'),
            "category": refined_cat
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
        opt_label_match = re.match(r'^([a-z])[\.\s]', opt.lower())
        opt_label = opt_label_match.group(1) if opt_label_match else opt.lower()

        # 如果这个选项是正确答案之一，给它加个绿色的框或加粗
        if opt_label in [a.lower() for a in q['answer']]:
            st.markdown(f":green[👉 **{opt}** (正确答案)]")
        else:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{opt}")

    # --- 这里的解析现在在答案揭晓后显示了 ---
    if q.get('explanation'):
        if q.get('explanation') not in [".", "。", "!", "！", "null", "None"] and len(q.get('explanation')) > 2:
            st.info(f"💡 **Buddy 的解析：** {q['explanation']}")
    else:
        None

    # 问题 1 修复：不展示习题块原文，展示回溯到的原理原文
    # --- 【按需触发：原理回溯】 ---
    with st.spinner("Buddy 正在翻阅手册原文..."):
        principle_text, principle_img = fetch_principle_context(q['metadata'], q['question'])

    with st.expander("📖 查看手册原理复盘"):
        if principle_text:
            if principle_img:
                st.image(principle_img.split(",")[0], caption="知识点关联图示")
            st.markdown(principle_text)
            st.caption(f"注：以上内容来自OW教材手册【{q['metadata'].get('Header1', '通用章节')}】")
        else:
            st.write("Buddy 暂时没在库里找到这段原理的直接原文，建议参考上方解析。")

    if "debug_probe" in st.session_state:
        st.caption(
            f"🧪 调试信息 -> 你选了: `{st.session_state.debug_probe['user_label']}` | 正确答案: `{st.session_state.debug_probe['correct_ans']}`")

    if st.button("下一题 ➡️", use_container_width=True, key=f"next_btn_{st.session_state.quiz_step}"):
        st.session_state.quiz_step += 1
        st.session_state.quiz_stage = "asking"
        st.rerun()

def render_summary():
    st.balloons()
    st.header("🏁 复习达成！这是你的表现报告")

    # 1. 顶部数据看板
    c1, c2, c3 = st.columns(3)
    total = len(st.session_state.current_quiz_set)
    correct = st.session_state.score_box["correct"]
    c1.metric("总题数", total)
    c2.metric("答对", correct)
    c3.metric("正确率", f"{(correct / total) * 100:.0f}%")

    st.divider()

    # 2. 错题集详细列表 (你的核心需求)
    if st.session_state.wrong_log:
        st.subheader("❌ 错题复盘 (哪里不会补哪里)")

        for i, item in enumerate(st.session_state.wrong_log):
            with st.container(border=True):
                st.markdown(f"**错题 {i + 1}:** {item['question']}")

                options_list = item.get('options', [])
                if options_list:
                    st.caption("选项回顾：")

                    # 准备比对用的标准列表（全部转小写、去空格）
                    correct_ids = [str(a).strip().lower() for a in item.get('correct_answer', [])]
                    user_ids = [str(a).strip().lower() for a in item.get('user_answer', [])]

                    for opt in options_list:
                        # --- 【核心修复：提取选项的唯一身份 ID】 ---
                        opt_raw = str(opt).strip().lower()
                        # 只有 A. B. 这种才提首字母，否则保留全文（如“正确”）
                        import re
                        label_match = re.match(r'^([a-z])[\.\s]', opt_raw)
                        opt_id = label_match.group(1) if label_match else opt_raw
                        # ------------------------------------------

                        is_correct = opt_id in correct_ids
                        is_user_selected = opt_id in user_ids

                        if is_correct:
                            # 如果是正确答案（不管你选没选），标绿并打勾
                            st.markdown(f"✅ :green[**{opt}** (正确答案)]")
                        elif is_user_selected:
                            # 如果是你选了，但它不是正确答案，标红并打叉
                            st.markdown(f"❌ :red[**{opt}** (你的选择)]")
                        else:
                            # 没选且错误的选项，灰度显示
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{opt}")

                # 展示解析
                if item['explanation']:
                    st.info(f"💡 **解析：** {item['explanation']}")
                st.write(f"🏷️ 归类: `{item['category']}`")

        # 3. AI 情感化总结 (Buddy 的最后叮嘱)
        st.write("---")
        st.subheader("🕵️ Buddy 的暖心建议")

        from config import QUIZ_SUMMARY_TEMPLATE
        from models import get_models
        _, agent_llm, _ = get_models()

        wrong_cats = list(set([it['category'] for it in st.session_state.wrong_log]))
        final_prompt = QUIZ_SUMMARY_TEMPLATE.format(
            correct_count=correct,
            wrong_count=st.session_state.score_box["wrong"],
            wrong_categories=wrong_cats,
            wrong_questions=[q['question'][:20] for q in st.session_state.wrong_log]
        )

        with st.spinner("Buddy 正在看你的成绩单..."):
            summary_res = agent_llm.invoke(final_prompt)
            st.write(summary_res.content)

    else:
        st.success("这波复习你拿了满分！看来你已经完全准备好下水了！")

    if st.button("🔄 开启新一轮挑战", type="primary", key="summary_retry"):
        st.session_state.current_quiz_set = []  # 触发重新加载
        st.rerun()