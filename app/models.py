# models.py
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import config
import os

# 初始化 RAG 组件 (LLM 、SLM、embedding和 ChromaDB)
# =======================================================

def _initialize_models():
    # 这一步非常关键：在初始化任何 LangChain 对象前，先设置环境变量
    if not config.SILICONFLOW_API_KEY:
        raise ValueError("SILICONFLOW_API_KEY 未在 config 中定义")

    # 强行设置环境变量，这是底层 OpenAI SDK 的“最后防线”
    os.environ["OPENAI_API_KEY"] = config.SILICONFLOW_API_KEY

    # 主agent大模型
    agent_llm = ChatOpenAI(
        model=config.RAG_LLM_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.0
    )
    # 知识和问题向量化的embedding模型
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL_CLOUD,
        api_key=config.SILICONFLOW_API_KEY,
        base_url=config.SILICON_BASE_URL,
    )
    # 用于意图识别、query改写、实体提取的小模型
    slm_parser = ChatOpenAI(
        model=config.SLM_MODEL,
        api_key=config.SILICONFLOW_API_KEY,
        base_url=config.SILICON_BASE_URL,
        temperature=0.1
    )
    # 加载向量数据库
    rag_db = Chroma(persist_directory=config.CHROMA_PATH, embedding_function=embeddings)

    return slm_parser, agent_llm, rag_db

@st.cache_resource
def get_models():
    """这是给 Streamlit 网页用的带缓存的壳"""
    try:
        return _initialize_models()
    except Exception as e:
        st.error(f"网页端模型加载失败: {e}")
        return None, None, None