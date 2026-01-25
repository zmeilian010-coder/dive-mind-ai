import os
import pandas as pd
import time
from dotenv import load_dotenv
import json
from pathlib import Path

# 从 openai 库导入 OpenAI 客户端 (仍然需要，因为 LangchainLLMWrapper 会用)
from openai import OpenAI

# RAGAS 评估框架
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
# 导入旧版本 RAGAS 的 LangChain LLM 和 Embedding 封装
from ragas.llms import LangchainLLMWrapper # <-- 回退到这个导入
from ragas.embeddings import LangchainEmbeddingsWrapper # <-- 回退到这个导入


# LangChain 组件 (用于调用你的RAG系统和评估模型)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings as LangchainHuggingFaceEmbeddings # 从 langchain_huggingface 导入
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 从 datasets 导入 Dataset 类
from datasets import Dataset

# 加载 .env 文件中的环境变量
load_dotenv()

# =======================================================
# 你的 RAG 系统配置 (与 query.py 保持一致)
# =======================================================
CHROMA_PATH = "chroma"
# RAG系统使用的Embedding模型.这个路径指向本地的 BGE-M3 模型文件
LOCAL_BGE_M3_MODEL_PATH = Path("E:/Python项目/dify应用的评估效果/local_bge_m3_model/bge-m3")
# RAG 系统使用的 Embedding 模型名称 (现在指向本地路径)
RAG_EMBEDDING_MODEL_NAME = str(LOCAL_BGE_M3_MODEL_PATH)
RAG_LLM_MODEL = os.getenv("LLM_MODEL")  # RAG系统使用的LLM模型
DEEPSEEK_BASE_URL = os.getenv("OPENAI_API_BASE")

# =======================================================
# RAGAS 评估器配置
# =======================================================
QWEN_API_KEY = os.getenv("QWEN_API_KEY")  # QWEN API Key
JUDGE_OPENAI_API_BASE_QWEN = os.getenv("JUDGE_OPENAI_API_BASE")  # QWEN API Base URL


# 其他配置
QA_DATASET_FILE = "数据评测/test_QA.csv"  # 你的问答数据集文件
EVALUATION_RESULT_FILE = "数据评测/evaluation result_test_QA.csv"  # 评估结果输出文件

# 新增：RAG 回复缓存文件
RAG_RESPONSES_CACHE_FILE = "数据评测/rag_responses_cache_2511042306.json"
# 新增：是否强制重新生成 RAG 回复 (True: 每次都重新生成，False: 如果文件存在则加载)
FORCE_REGENERATE_RAG_RESPONSES = False # 调试时设为 True，稳定后设为 False

# 定义 RAG 问答的提示模板 (与 query.py 保持一致)
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一个智能问答助手，请根据提供的上下文信息，准确地回答问题。如果上下文中没有提到，请明确表示不知道。"),
        ("human", "上下文: {context}\n\n问题: {question}")
    ]
)


def format_docs(docs: list) -> str:
    """
    将检索到的文档列表格式化成一个字符串，方便作为上下文传入 LLM。
    """
    return "\n\n".join(doc.page_content for doc in docs)


# --- RAGS 评估的主要函数 ---
def run_ragas_evaluation():
    # 声明 FORCE_REGENERATE_RAG_RESPONSES 是一个全局变量
    global FORCE_REGENERATE_RAG_RESPONSES
    # --- 1. 初始化 RAG 系统组件 ---
    print("--- 1. 初始化 RAG 系统组件 (用于生成回答和上下文) ---")
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。请在 .env 文件中配置。")

    # RAG 系统自身的 Embedding 模型，用于从 ChromaDB 检索
    rag_embeddings = LangchainHuggingFaceEmbeddings(
        model=RAG_EMBEDDING_MODEL_NAME,  # <--- 这里会使用你的本地路径字符串
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
    )

    if not os.path.exists(CHROMA_PATH):
        raise ValueError(f"ChromaDB 路径 '{CHROMA_PATH}' 不存在。请确保已运行 ingest.py 创建了知识库。")

    rag_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag_embeddings)
    rag_retriever = rag_db.as_retriever(search_kwargs={"k": 10})  # k=3 检索3个最相关文档块

    # RAG 系统自身的 LLM (DeepSeek)
    rag_llm = ChatOpenAI(
        model=RAG_LLM_MODEL,
        openai_api_base=DEEPSEEK_BASE_URL,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.0
    )

    # 构建 RAG Chain (用于生成答案和获取上下文)
    rag_chain = (
            {"context": rag_retriever | RunnableLambda(lambda docs: docs),
             "question": RunnablePassthrough()}
            | RAG_PROMPT_TEMPLATE
            | rag_llm
            | StrOutputParser()
    )

    print("RAG 系统组件初始化完成。")

    # --- 2. 初始化 RAGAS 评估模型 (使用 QWEN 和 BGE-M3) ---
    print("\n--- 2. 正在初始化 RAGAS 评估模型 (LLM: QWEN, Embeddings: BGE-M3) ---")

    # 检查 QWEN API Key 和 Base URL
    if not QWEN_API_KEY:
        raise ValueError("QWEN_API_KEY 环境变量未设置。请在 .env 文件中配置。")
    if not JUDGE_OPENAI_API_BASE_QWEN:
        raise ValueError("OPENAI_API_BASE (QWEN) 环境变量未设置。请在 .env 文件中配置。")

    try:
        # LLM Evaluator (DS)
        evaluator_llm_langchain = ChatOpenAI(
            model="qwen-flash",
            temperature=0.2,
            openai_api_base=JUDGE_OPENAI_API_BASE_QWEN,
            openai_api_key=QWEN_API_KEY,
            n=3,
            extra_body={"enable_thinking": False},
            request_timeout=240,
        )
        evaluator_llm = LangchainLLMWrapper(evaluator_llm_langchain)

        # Embedding Evaluator (BGE-M3) - 使用 LangchainEmbeddingsWrapper
        evaluator_embeddings_langchain = LangchainHuggingFaceEmbeddings(  # <-- 注意这里依然是 LangchainHuggingFaceEmbeddings
            model_name=RAG_EMBEDDING_MODEL_NAME,  # <-- 这里是 model_name
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # <-- 重新添加 encode_kwargs
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            evaluator_embeddings_langchain)  # <-- 使用 LangchainEmbeddingsWrapper
        print("RAGAS 评估模型初始化完成。")

    except Exception as e:
        print(f"ERROR: RAGAS 评估模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 为 RAGAS 注册 LLM 和 Embeddings (确保它们在 RAGAS 内部被正确使用)
    # 这一步非常重要，Ragas 依赖于此来使用你提供的 LLM 和 Embedding 模型进行评估
    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    context_recall.llm = evaluator_llm
    context_precision.llm = evaluator_llm

    print("RAGAS 评估指标模型注册完成。")

    # Answer Relevancy 和 Context Relevancy 也用 embedding
    # 如果你也想评估 AnswerRelevancy 和 ContextRelevancy 中的embedding，需要设置
    answer_relevancy.embeddings = evaluator_embeddings  # Answer Relevancy 可能也需要 embeddings
    # context_relevancy.embeddings = evaluator_embeddings # Context Relevancy 也可能需要 embeddings

    print("RAGAS 评估指标模型注册完成。")

    # --- 3. 加载数据集 ---
    print(f"\n--- 3. 正在加载问答数据集: '{QA_DATASET_FILE}' ---")
    if not os.path.exists(QA_DATASET_FILE):
        raise ValueError(f"问答数据集文件 '{QA_DATASET_FILE}' 不存在。请确保已生成。")

    try:
        # 你的 CSV 文件编码是 gbk
        df = pd.read_csv(QA_DATASET_FILE, encoding="gbk")
        df['question'] = df['question'].astype(str)
        df['ground_truth'] = df['ground_truth'].astype(str)
        questions = df["question"].tolist()
        ground_truths = df["ground_truth"].tolist()
        print(f"成功加载 {len(questions)} 对问答数据。")
    except Exception as e:
        print(f"ERROR: 加载问答数据集失败: {e}")
        return

    # --- 4. 获取 RAG 系统生成答案和上下文 (从缓存或实时生成) ---
    print("\n--- 4. 正在获取 RAG 系统生成答案和上下文 ---")
    cached_data = {"answers": [], "contexts": []}

    # 检查是否存在缓存文件
    if os.path.exists(RAG_RESPONSES_CACHE_FILE) and not FORCE_REGENERATE_RAG_RESPONSES:
        print(f"检测到缓存文件 '{RAG_RESPONSES_CACHE_FILE}'，正在加载...")
        try:
            with open(RAG_RESPONSES_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            # 简单校验数据量是否匹配
            if len(cached_data["answers"]) == len(questions) and len(cached_data["contexts"]) == len(questions):
                answers = cached_data["answers"]
                contexts = cached_data["contexts"]
                print(f"成功从缓存加载 {len(answers)} 对 RAG 回复。")
            else:
                print("缓存文件数据量不匹配，将重新生成 RAG 回复。")
                FORCE_REGENERATE_RAG_RESPONSES = True  # 强制重新生成
        except Exception as e:
            print(f"!!! 警告：加载缓存文件 '{RAG_RESPONSES_CACHE_FILE}' 失败：{e}。将重新生成 RAG 回复。")
            FORCE_REGENERATE_RAG_RESPONSES = True

    if FORCE_REGENERATE_RAG_RESPONSES or not cached_data["answers"]:
        print("正在调用 RAG 系统为每个问题生成答案和检索上下文...")
        answers = []
        contexts = []

        for i, q in enumerate(questions):
            print(f"  - 处理问题 {i + 1}/{len(questions)}: '{q[:50]}...'")
            try:
                retrieved_docs = rag_retriever.invoke(q)
                context_list_for_rag = [doc.page_content for doc in retrieved_docs]
                context_string_for_llm = format_docs(retrieved_docs)

                # 确保 DeepSeek API 请求中没有 n 参数，或设置为 1，避免冲突
                rag_response = rag_llm.invoke(RAG_PROMPT_TEMPLATE.format_messages(
                    context=context_string_for_llm,
                    question=q
                ))

                answers.append(rag_response.content)
                contexts.append(context_list_for_rag)

            except Exception as e:
                print(f"!!! 错误：RAG 系统处理问题 '{q[:50]}...' 失败：{e}")
                answers.append("ERROR: RAG system failed to generate answer.")
                contexts.append([])  # 如果失败，上下文为空

            time.sleep(0.5)  # 添加一个小的延迟

        # 保存 RAG 回复到缓存文件
        print(f"RAG 回复生成完成，正在保存到缓存文件 '{RAG_RESPONSES_CACHE_FILE}'...")
        try:
            with open(RAG_RESPONSES_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({"answers": answers, "contexts": contexts}, f, ensure_ascii=False, indent=2)
            print("RAG 回复缓存保存成功。")
        except Exception as e:
            print(f"!!! 警告：保存 RAG 回复缓存失败：{e}")

    # --- 5. 组装 RAGAS 数据集 ---
    print("\n--- 5. 组装 RAGAS 数据集 ---")
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": ground_truths,
    }

    ragas_dataset = Dataset.from_dict(data)

    print("RAGAS 数据集组装完成。")

    # --- 6. 运行 RAGAS 评估 ---
    print("\n--- 6. 正在运行 RAGAS 评估 ---")
    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    try:
        # RAGAS 评估的全局超时
        result = evaluate(
            ragas_dataset,
            metrics=metrics_to_evaluate,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            # RAGAS 0.0.x 版本的超时和并发参数
            # max_concurrency=2, # 可以尝试调整并发度
            # timeout=180, # 设置一个更长的全局超时时间 (秒)
        )
        print("\n--- RAGAS 评估结果 ---")
        print(result)

        result_df = result.to_pandas()
        result_df.to_csv(EVALUATION_RESULT_FILE, index=False, encoding="utf-8-sig")
        print(f"详细评估结果已保存到 '{EVALUATION_RESULT_FILE}'。")

    except Exception as e:
        print(f"ERROR: 运行 RAGAS 评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("WARN: DEEPSEEK_API_KEY 环境变量未设置，RAG系统可能无法正常工作。")
    if not QWEN_API_KEY:
        print("WARN: QWEN_API_KEY 环境变量未设置，RAGAS评估器可能无法正常工作。")
    if not JUDGE_OPENAI_API_BASE_QWEN:
        print("WARN: OPENAI_API_BASE (QWEN) 环境变量未设置，QWEN LLM可能无法正常工作。")

    run_ragas_evaluation()