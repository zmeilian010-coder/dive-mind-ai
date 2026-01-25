# models.py
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import config


# 初始化 RAG 组件 (LLM 、SLM、embedding和 ChromaDB)
# =======================================================

@st.cache_resource
def get_models():
    try:
        # 主agent大模型
        agent_llm = ChatOpenAI(
            model=config.RAG_LLM_MODEL,
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=0.0
        )
        # 知识和问题向量化的embedding模型
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_CLOUD,
            api_key=st.secrets["SILICONFLOW_API_KEY"],
            base_url=config.SILICON_BASE_URL,
        )
        # 用于意图识别、query改写、实体提取的小模型
        slm_parser = ChatOpenAI(
            model=config.SLM_MODEL,
            api_key=st.secrets["SILICONFLOW_API_KEY"],
            base_url=config.SILICON_BASE_URL,
            temperature=0.1
        )
        # 加载向量数据库
        rag_db = Chroma(persist_directory=config.CHROMA_PATH, embedding_function=embeddings)

        # 【检查点】确保这一行存在，且在 try 块的最后！
        return slm_parser, agent_llm, rag_db

    except Exception as e:
        st.error(f"模型初始化失败，请检查密钥和路径: {e}")
        # 如果报错，也得返回三个 None，防止外部解包失败
        return None, None, None