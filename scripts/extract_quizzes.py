import os
import re
import json
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any
# 获取根目录
current_file = Path(__file__).resolve()
root_path = current_file.parent.parent
sys.path.append(str(root_path))

# ！！！先导入 config，确保环境变量被加载 ！！！
import config
# 然后再导入 models
from app import models # 注意这里加了 app. 前缀
# =======================================================
# 🚀 路径导航：解决跨文件夹导入问题
# =======================================================

# 现在可以正常导入了
from langchain_core.documents import Document


# =======================================================
# 核心功能逻辑
# =======================================================

def generate_stable_id(parent_id: str, text: str) -> str:
    """根据父块ID和题目内容生成稳定的唯一ID"""
    content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"qz_{parent_id[:6]}_{content_hash}"


def parse_single_question(text: str, parent_id: str, metadata: dict) -> Dict[str, Any]:
    """
    核心解析逻辑：从一段文本中通过正则提取所有结构化字段
    """
    raw_text = text.strip()

    # 1. 提取题干 (匹配数字开头到第一个 □ 之前)
    q_match = re.search(r'^\d+\.(.*?)(?=□)', raw_text, re.DOTALL)
    question_text = q_match.group(1).strip() if q_match else "未知题干"

    # 2. 提取选项 (匹配所有 □ 开头的行)
    options = re.findall(r'□\s*(.*)', raw_text)
    clean_options = [opt.strip() for opt in options]

    # 3. 提取答案与解析
    ans_pattern = r'- 题目 \d+ 答案:\s*([a-zA-Z,，\s正确错误\.]+)(.*)'
    ans_match = re.search(ans_pattern, raw_text, re.DOTALL)

    raw_answer = ""
    explanation = ""
    if ans_match:
        raw_answer = ans_match.group(1).strip().rstrip('.')
        explanation = ans_match.group(2).strip()

        # 4. 判定题型
    ans_list = []
    q_type = "single"

    if "正确" in clean_options or "错误" in clean_options:
        q_type = "true_false"
        ans_list = [raw_answer]
    else:
        potential_chars = re.findall(r'[a-zA-Z]', raw_answer)
        ans_list = [c.lower() for c in potential_chars]
        if len(ans_list) > 1:
            q_type = "multiple"
        else:
            q_type = "single"

    return {
        "question_id": generate_stable_id(parent_id, question_text),
        "parent_chunk_id": parent_id,
        "question": question_text,
        "options": clean_options,
        "answer": ans_list,
        "type": q_type,
        "explanation": explanation if explanation else None,
        "raw_text": raw_text,
        "metadata": metadata     # 继承母chunk的元数据
    }


def run_extraction():
    print(f"🚀 正在连接数据库: {config.CHROMA_PATH}")

    # --- 增加这段诊断代码 ---
    print(f"DEBUG: 根目录路径是 {config.ROOT_DIR}")
    print(f"DEBUG: 硅基流动 Key 长度: {len(config.SILICONFLOW_API_KEY) if config.SILICONFLOW_API_KEY else '空的'}")

    # 1. 获取数据库实例 (从 models 工厂拿)
    # 注意：在 scripts 中运行，config 里的 EMBEDDING_MODE 需与数据库一致
    _, _, rag_db = models._initialize_models()
    print(f"DEBUG: rag_db 是否为空: {rag_db is None}") # 看看这行输出什么

    # 2. 抓取所有习题块
    raw_data = rag_db.get(where={"is_quiz": True}, include=['documents', 'metadatas'])

    total_chunks = len(raw_data['ids'])
    if total_chunks == 0:
        print("❌ 错误：未发现 is_quiz=True 的数据。请检查 config 里的路径。")
        return

    print(f"📦 发现 {total_chunks} 个习题文本块，正在拆解为原子题目...")

    all_parsed_questions = []

    for i in range(total_chunks):
        content = raw_data['documents'][i]
        meta = raw_data['metadatas'][i]
        chunk_id = raw_data['ids'][i]

        # 按照“换行+数字点”切分
        items = re.split(r'\n(?=\d+\.)', content.strip())

        for q_text in items:
            if "答案:" in q_text:
                try:
                    parsed_q = parse_single_question(q_text, chunk_id, meta)
                    all_parsed_questions.append(parsed_q)
                except Exception as e:
                    print(f"⚠️ 解析跳过: {e}")

    # 3. 保存结果
    output_path = os.path.join(config.DATA_DIR, "processed", "quiz_bank.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_parsed_questions, f, ensure_ascii=False, indent=4)

    print(f"✅ 成功提取 {len(all_parsed_questions)} 道题目！文件已保存。")


if __name__ == "__main__":
    run_extraction()