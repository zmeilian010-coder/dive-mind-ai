import os
import shutil
import time
import sys
from pathlib import Path

# 确保能导入 config
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

import config
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb


def run_diagnostic():
    print("=== 🛠️ DiveMind 数据库链路全诊断开始 ===")

    # 环节 1: 环境检查
    print(f"\n1. [环境检查]")
    print(f"   - Python 版本: {sys.version}")
    try:
        import chromadb
        print(f"   - ChromaDB 版本: {chromadb.__version__}")
    except:
        print(f"   - ❌ 未检测到 chromadb 库")

    # 环节 2: Embedding 引擎检查 (关键！)
    print(f"\n2. [Embedding 引擎测试]")
    try:
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_CLOUD,
            api_key=st.secrets["SILICONFLOW_API_KEY"],
            base_url=config.SILICON_BASE_URL
        )
        test_vector = embeddings.embed_query("测试文字")
        print(f"   - ✅ 成功获取向量！维度: {len(test_vector)}")
        print(f"   - 向量前 3 位样例: {test_vector[:3]}")
    except Exception as e:
        print(f"   - ❌ Embedding 引擎连接失败: {e}")
        return

    # 环节 3: 物理路径与权限检查
    print(f"\n3. [物理路径检查]")
    test_path = os.path.join(root_path, "debug_db_test")
    print(f"   - 测试路径: {test_path}")
    if os.path.exists(test_path):
        try:
            shutil.rmtree(test_path)
            print("   - ✅ 旧测试文件夹清理成功")
        except Exception as e:
            print(f"   - ❌ 无法删除旧文件夹 (可能被占用): {e}")
            return
    os.makedirs(test_path, exist_ok=True)
    print("   - ✅ 文件夹创建/写入权限正常")

    # 环节 4: 原生 Chroma 写入测试 (不通过 LangChain)
    print(f"\n4. [原生 Chroma 写入测试]")
    try:
        client = chromadb.PersistentClient(path=test_path)
        collection = client.create_collection(name="test_collection")
        collection.add(
            ids=["id1"],
            embeddings=[test_vector],
            documents=["这是一条测试内容"]
        )
        print(f"   - ✅ 原生客户端写入成功")
        print(f"   - 库内条数: {collection.count()}")
        client = None  # 尝试释放
        time.sleep(1)
    except Exception as e:
        print(f"   - ❌ 原生写入失败: {e}")

    # 环节 5: 索引落盘检查
    print(f"\n5. [索引文件落盘检查]")
    # 查找是否有子文件夹生成
    subdirs = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    print(f"   - 数据库子文件夹: {subdirs}")
    if subdirs:
        index_files = os.listdir(os.path.join(test_path, subdirs[0]))
        print(f"   - 索引内容文件: {index_files}")
        has_bin = any(f.endswith('.bin') for f in index_files)
        print(f"   - ✅ 是否包含 .bin 文件: {has_bin}")
    else:
        print("   - ❌ 警告：未发现 HNSW 索引文件夹")

    # 环节 6: LangChain 兼容性二次加载测试
    print(f"\n6. [LangChain 加载测试]")
    try:
        db = Chroma(
            persist_directory=test_path,
            embedding_function=embeddings,
            collection_name="test_collection"
        )
        res = db.similarity_search("测试", k=1)
        print(f"   - ✅ 加载成功！检索结果: {res[0].page_content}")
    except Exception as e:
        print(f"   - ❌ LangChain 加载失败: {e}")

    print("\n" + "=" * 40)
    print("诊断结束，请查看上方每一步的输出。")


if __name__ == "__main__":
    run_diagnostic()