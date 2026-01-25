import os
import shutil
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredExcelLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path
import hashlib
import json
import datetime
from collections import defaultdict

load_dotenv()

# =======================================================
# 路径配置
# =======================================================
KNOWLEDGE_BASE_DIR = "docs"
DATA_PATH = KNOWLEDGE_BASE_DIR
CHROMA_PATH = "chroma"
DOCUMENT_STATUS_FILE = "document_status.json"  # 用于记录文档状态的文件

LOCAL_BGE_M3_MODEL_PATH = Path("E:/Python项目/dify应用的评估效果/local_bge_m3_model/bge-m3")
RAG_EMBEDDING_MODEL_NAME = str(LOCAL_BGE_M3_MODEL_PATH)

# =======================================================
# 自定义元数据设置
# =======================================================
DEFAULT_PROJECT_NAME = "DiveKnowledgeBase"
MARKDOWN_DOC_VERSION = "1.0"

EXTERNAL_METADATA_EXCEL = "chunks_with_category.xlsx"  # 外部标注Excel文件名
EXTERNAL_METADATA_CHUNK_ID_COL = "Chunk_ID_Hash"
EXTERNAL_METADATA_CATEGORY_COL = "Category"

# =======================================================
# --- 配置模式选择 ---
# 设置为 True 进行增量更新 (只处理修改过的文件和新文件)
# 设置为 False 进行全量重建 (清空数据库，从头开始)
INCREMENTAL_UPDATE_DEFAULT = True  # <-- 在这里修改 True 或 False 来切换模式
# =======================================================

# =======================================================
# 定义需要特殊处理的元数据字段及其目标类型
# =======================================================
FORCE_STR_KEYS = {
    'boatId', 'tourId', 'tripId', 'experience', 'certification',
    'nameCN', 'nameEN', 'locationName', 'diving_equipment', 'languages', 'policy', 'nitrox', 'wifi',
    'tech_diving_friendly',
    'chunk_type', 'category', 'Metadata_source', 'Metadata_file_type', 'Metadata_Header1', 'Metadata_Header2',
    'Metadata_row_number',  # Excel的行号也作为字符串
}
FORCE_FLOAT_KEYS = {'rating'}
FORCE_INT_KEYS = {'yearBuilt', 'dives', 'duration', 'nights'}
BOOLEAN_KEYS = {}
DATE_TIME_STR_KEYS = {'arrivalDate', 'departureDate', 'updatedTime'}


# =======================================================
# 辅助函数
# =======================================================
def load_document_status():
    """加载上次运行的文档处理状态"""
    if os.path.exists(DOCUMENT_STATUS_FILE):
        with open(DOCUMENT_STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_document_status(status):
    """保存当前文档处理状态"""
    with open(DOCUMENT_STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def _normalize_metadata_value(col_name, value):
    """辅助函数：对单个元数据值进行类型转换和规范化"""
    if pd.isna(value) or (isinstance(value, str) and value.strip() == ''):  # 统一处理 NaN 和空字符串
        return None

    str_value = str(value).strip()

    if col_name in FORCE_STR_KEYS:
        return str_value
    elif col_name in FORCE_FLOAT_KEYS:
        try:
            return float(str_value)
        except ValueError:
            return None
    elif col_name in FORCE_INT_KEYS:
        try:
            return int(float(str_value))
        except ValueError:
            return None
    elif col_name in BOOLEAN_KEYS:
        lower_val = str_value.lower()
        if lower_val in ('true', 'yes'):
            return True
        elif lower_val in ('false', 'no'):
            return False
        else:
            return None
    elif col_name in DATE_TIME_STR_KEYS:
        try:
            if isinstance(value, datetime.datetime):  # 使用 datetime.datetime
                return value.isoformat()
            else:
                dt_obj = pd.to_datetime(str_value)
                return dt_obj.isoformat()
        except (ValueError, TypeError):
            return None
    else:
        return str_value


def generate_chunk_hash(chunk: Document) -> str:
    """为文档块生成唯一的哈希ID"""
    unique_string_parts = {
        "page_content": chunk.page_content,
        # 显式列出所有可能影响哈希值的元数据字段
        "source": chunk.metadata.get("source"),
        "file_type": chunk.metadata.get("file_type"),
        "row_number": chunk.metadata.get("row_number"),
        "Header1": chunk.metadata.get("Header1"),
        "Header2": chunk.metadata.get("Header2"),
        "boatId": chunk.metadata.get("boatId"),
        "tourId": chunk.metadata.get("tourId"),
        "tripId": chunk.metadata.get("tripId"),
        "nameCN": chunk.metadata.get("nameCN"),
        "nameEN": chunk.metadata.get("nameEN"),
        "locationName": chunk.metadata.get("locationName"),
        "arrivalDate": chunk.metadata.get("arrivalDate"),
        "departureDate": chunk.metadata.get("departureDate"),
        "updatedTime": chunk.metadata.get("updatedTime"),
        "experience": chunk.metadata.get("experience"),
        "certification": chunk.metadata.get("certification"),
        "dives": chunk.metadata.get("dives"),
        "duration": chunk.metadata.get("duration"),
        "nights": chunk.metadata.get("nights"),
        "nitrox": chunk.metadata.get("nitrox"),
        "wifi": chunk.metadata.get("wifi"),
        "diving_equipment": chunk.metadata.get("diving_equipment"),
        "tech_diving_friendly": chunk.metadata.get("tech_diving_friendly"),
        "languages": chunk.metadata.get("languages"),
        "policy": chunk.metadata.get("policy"),
        "rating": chunk.metadata.get("rating"),
        "yearBuilt": chunk.metadata.get("yearBuilt"),
        "chunk_type": chunk.metadata.get("chunk_type"),
        "category": chunk.metadata.get("category"),
        "project": chunk.metadata.get("project"),
        "version": chunk.metadata.get("version"),
        "processed_by": chunk.metadata.get("processed_by"),
        "original_source": chunk.metadata.get("original_source"),
    }
    # 将所有非None的值转换为字符串再进行哈希计算
    unique_string_parts_cleaned = {k: str(v) for k, v in unique_string_parts.items() if v is not None}
    sorted_unique_string = json.dumps(unique_string_parts_cleaned, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(sorted_unique_string.encode('utf-8')).hexdigest()


def create_database(incremental_update: bool = INCREMENTAL_UPDATE_DEFAULT):  # 使用顶部定义的默认值
    print(f"--- [{datetime.datetime.now()}] 步骤 1: 加载文档 '{KNOWLEDGE_BASE_DIR}' 中的所有支持文件 ---")

    loader_map = {
        ".xlsx": UnstructuredExcelLoader,
        ".txt": TextLoader,
        # .json 文件现在是手动读取和处理，不再使用 LangChain Loader
    }

    # 读取旧的文档处理状态，用于增量更新
    current_document_status = load_document_status()
    new_document_status = {}  # 记录本次处理后的新状态

    # 初始化 Embedding 模型 (BGE-M3) 一次
    print(f"--- [{datetime.datetime.now()}] 步骤 3: 初始化 Embedding 模型 (BGE-M3) ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=RAG_EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
    )
    print(f"正在使用 BGE-M3 模型 '{RAG_EMBEDDING_MODEL_NAME}' 生成 embeddings...")
    print("首次运行时，模型会自动从 Hugging Face 下载。请耐心等待。")


    # --- 全量更新逻辑修正 ---
    db_instance = None # 声明 db_instance 但不立即初始化
    if not incremental_update:  # 如果不是增量更新，就是全量重建
        if os.path.exists(CHROMA_PATH):
            print(f"--- [{datetime.datetime.now()}] 全量重建模式：正在删除旧的 ChromaDB 文件夹: {CHROMA_PATH} ---")
            # 确保在初始化 Chroma 实例之前执行删除操作
            shutil.rmtree(CHROMA_PATH)
            print("旧的 ChromaDB 文件夹已删除。")
        else:
            print("--- 全量重建模式：ChromaDB 文件夹不存在，将新建。 ---")

        # 此时才初始化一个全新的 Chroma 实例，确保它操作的是一个不存在或刚被删除的目录
        db_instance = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        current_document_status = {}  # 全量重建清空旧状态
        print("新的 ChromaDB 已初始化，开始重建。")
    else:  # 增量更新模式
        print("--- 增量更新模式：将检查文档修改并更新。 ---")
        # 增量更新模式下，ChromaDB 必须存在。初始化 Chroma 实例来打开现有数据库。
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(
                f"ChromaDB 路径 '{CHROMA_PATH}' 不存在。增量更新需要一个现有的数据库。请先进行全量重建。")

        # 增量模式下，初始化 Chroma 实例来打开现有数据库
        db_instance = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        print("现有 ChromaDB 已加载。")

    all_chunks_for_processing = []  # 存储从所有文件加载和分块后的原始 chunks
    processed_file_paths_current_run = set()  # 记录本次循环实际处理过的文件路径，用于文件状态更新

    # 遍历 docs 文件夹加载和分块文档
    for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
        # 移除此行，允许处理子目录。如果只希望处理顶层，请恢复。
        # if Path(root) != Path(KNOWLEDGE_BASE_DIR): continue

        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1].lower()
            file_mtime = os.path.getmtime(file_path)  # 获取文件修改时间戳

            # 跳过特定文件
            if file_extension == ".pdf": continue
            if file_path.startswith(CHROMA_PATH + os.sep) or file_path == CHROMA_PATH: continue
            if Path(file_path).name.lower() == Path(EXTERNAL_METADATA_EXCEL).name.lower(): continue

            # 使用文件的相对路径作为唯一文件ID，支持子目录
            relative_file_id = str(Path(file_path).relative_to(KNOWLEDGE_BASE_DIR))

            # --- 增量更新：检查文件是否需要重新处理 ---
            if incremental_update:
                last_mtime = current_document_status.get(relative_file_id, {}).get('last_mtime')
                if last_mtime == file_mtime:
                    print(
                        f"文件 '{file_path}' 未修改 ({datetime.datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')})，跳过处理。")
                    # 如果文件未修改，将其旧状态复制到新状态中，保留其 chunk_hashes
                    new_document_status[relative_file_id] = current_document_status[relative_file_id]
                    processed_file_paths_current_run.add(relative_file_id)  # 标记为已处理，不认为是删除
                    continue  # 跳过未修改的文件

            print(f"处理文件 '{file_path}' ({'重新处理' if incremental_update else '全量重建'}) ...")
            processed_file_paths_current_run.add(relative_file_id)  # 标记此文件被处理

            common_metadata = {
                "source": file_path,  # 完整文件路径
                "file_type": file_extension.lstrip('.'),
                "project": DEFAULT_PROJECT_NAME,
                "timestamp": file_mtime  # 记录原始文件时间戳，float
            }

            file_chunks = []  # 存储当前文件生成的所有chunk

            # --- Markdown 文件处理 ---
            if file_extension == ".md":
                original_pdf_name = Path(file_path).stem + ".pdf"
                original_pdf_path = Path(KNOWLEDGE_BASE_DIR) / original_pdf_name
                if original_pdf_path.exists():
                    common_metadata["processed_by"] = "PaddleOCR"
                    common_metadata["original_source"] = str(original_pdf_path)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                    md_specific_metadata = common_metadata.copy()
                    md_specific_metadata["version"] = MARKDOWN_DOC_VERSION

                    # 规范化Markdown文件自身的元数据
                    for k, v in list(md_specific_metadata.items()):  # 迭代副本，允许修改原字典
                        md_specific_metadata[k] = _normalize_metadata_value(k, v)
                    md_specific_metadata = {k: v for k, v in md_specific_metadata.items() if v is not None}  # 移除None

                    headers_to_split_on = [("#", "Header1"), ("##", "Header2")]
                    markdown_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                    md_chunks_initial = markdown_header_splitter.split_text(markdown_content)

                    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,
                                                                        length_function=len)

                    for chunk_initial in md_chunks_initial:
                        chunk_initial.metadata.update(md_specific_metadata)  # 合并已规范化的元数据
                        # 对Markdown块的元数据进行最终规范化 (如果解析器额外提取了什么)
                        for k, v in list(chunk_initial.metadata.items()):
                            chunk_initial.metadata[k] = _normalize_metadata_value(k, v)
                        chunk_initial.metadata = {k: v for k, v in chunk_initial.metadata.items() if v is not None}
                        file_chunks.extend(recursive_splitter.split_documents([chunk_initial]))

                except Exception as e:
                    print(f"!!! 警告：处理 Markdown 文件 '{file_path}' 失败：{e}")

            # --- XLSX, TXT 文件处理 ---
            elif file_extension in loader_map:
                try:
                    if file_extension == ".xlsx":
                        df = pd.read_excel(file_path)
                        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,
                                                                            length_function=len)
                        for index, row in df.iterrows():
                            row_content_parts = []
                            row_metadata_temp = common_metadata.copy()
                            row_metadata_temp["row_number"] = index + 1  # Excel行号也进行规范化，确保为str
                            row_metadata_temp["row_number"] = _normalize_metadata_value("Metadata_row_number",
                                                                                        row_metadata_temp["row_number"])

                            for col_name, value in row.items():
                                normalized_value = _normalize_metadata_value(col_name, value)
                                if normalized_value is not None:
                                    row_metadata_temp[col_name] = normalized_value
                                    row_content_parts.append(f"{col_name}: {normalized_value}")

                            row_content = ", ".join(row_content_parts) if row_content_parts else ""
                            row_metadata_temp = {k: v for k, v in row_metadata_temp.items() if v is not None}  # 移除None
                            file_chunks.extend(recursive_splitter.split_documents(
                                [Document(page_content=row_content, metadata=row_metadata_temp)]))
                    else:  # .txt 等使用 LangChain Loader
                        loader_cls = loader_map[file_extension]
                        loader = loader_cls(file_path)
                        docs_from_file = loader.load()
                        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,
                                                                            length_function=len)
                        for doc_initial in docs_from_file:
                            doc_initial.metadata.update(common_metadata)  # 合并通用元数据
                            # 对所有元数据进行最终规范化
                            for k, v in list(doc_initial.metadata.items()):
                                doc_initial.metadata[k] = _normalize_metadata_value(k, v)
                            doc_initial.metadata = {k: v for k, v in doc_initial.metadata.items() if v is not None}
                            file_chunks.extend(recursive_splitter.split_documents(
                                [Document(page_content=doc_initial.page_content,
                                          metadata=doc_initial.metadata)]))  # Fix: pass Document object
                except Exception as e:
                    print(f"!!! 警告：处理文件 '{file_path}' 失败：{e}")

            # --- JSON 文件手动读取和处理 (作为分块后的 Document 列表) ---
            elif file_extension == ".json":  # 新增 JSON 文件加载逻辑
                print(f"正在加载 JSON 知识文件: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

                    items_to_process = json_data if isinstance(json_data, list) else [json_data] if isinstance(
                        json_data, dict) else []

                    for item in items_to_process:
                        if "page_content" in item and "metadata" in item:  # 确保符合 Document 结构
                            doc_content = item["page_content"]
                            doc_metadata = item["metadata"].copy()

                            # 合并 common_metadata，但 JSON 文件中的优先级更高
                            for k, v in common_metadata.items():
                                if k not in doc_metadata:
                                    doc_metadata[k] = v

                            # 对JSON文件中的元数据进行规范化
                            for k, v in list(doc_metadata.items()):
                                doc_metadata[k] = _normalize_metadata_value(k, v)
                            doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}  # 移除None

                            json_doc = Document(page_content=doc_content, metadata=doc_metadata)
                            file_chunks.append(json_doc)  # 直接作为 chunk 添加，因为JSON已经分块好了
                        else:
                            print(
                                f"警告：JSON 文件 '{file_path}' 中有不符合 Document 格式的条目 (缺少 'page_content' 或 'metadata')。")
                except json.JSONDecodeError as e:
                    print(f"!!! 警告：JSON 文件 '{file_path}' 解析失败：{e}")
                except Exception as e:
                    print(f"!!! 警告：加载 JSON 文件 '{file_path}' 失败：{e}")

            else:
                print(f"!!! 警告：跳过不支持的文件类型: {file_path}")

            all_chunks_for_processing.extend(file_chunks)  # 将当前文件生成的所有 chunk 加入总列表

    if not all_chunks_for_processing:
        print(f"!!! 错误：处理后没有生成任何文本块。请检查文档内容或分割器配置。")
        return  # 提前退出，以便你检查上游问题

    print(f"--- [{datetime.datetime.now()}] 步骤 2: 文本分割、去重与哈希生成 ---")
    chunks = all_chunks_for_processing

    # =======================================================
    # DEBUG: 检查 chunks 列表中的元素类型 (新增)
    # =======================================================
    print(f"\n--- [{datetime.datetime.now()}] DEBUG: 检查文档块类型... ---")
    problematic_chunk_found = False
    for i, chunk_item in enumerate(chunks):
        if not isinstance(chunk_item, Document):
            print(f"!!! 错误：chunks 列表中的第 {i} 个元素不是 Document 类型，而是 {type(chunk_item)}: {chunk_item}")
            problematic_chunk_found = True
    if problematic_chunk_found:
        print("!!! 错误：检测到非 Document 类型的文档块，程序可能无法正常继续。请检查文本分割逻辑。")
        return  # 提前退出，以便你检查上游问题

    # --- 步骤 2.1: 基于 tourId 进行去重 (只保留第一个) ---
    final_deduplicated_chunks = []
    tour_chunks_map = defaultdict(list)
    total_chunks_before_tourid_dedup = len(chunks)

    print(f"处理前总文档块数 (包括临时重复): {total_chunks_before_tourid_dedup}")

    for chunk in chunks:
        tour_id = chunk.metadata.get("tourId")
        # 假设只有 '船宿路线' 类型或 '船宿行程' 类型需要 tourId 去重
        if tour_id is not None and chunk.metadata.get("chunk_type") in ["船宿路线", "船宿行程"]:
            tour_chunks_map[tour_id].append(chunk)
        else:
            final_deduplicated_chunks.append(chunk)

    for tour_id, chunk_list in tour_chunks_map.items():
        if len(chunk_list) > 1:
            print(f"警告：发现 tourId='{tour_id}' 有 {len(chunk_list)} 个重复文档块。只保留第一个。")
            final_deduplicated_chunks.append(chunk_list[0])
        else:
            final_deduplicated_chunks.extend(chunk_list)

    chunks = final_deduplicated_chunks
    chunks_after_tourid_dedup = len(chunks)
    print(f"基于 tourId 处理后，最终得到 {chunks_after_tourid_dedup} 个文档块。")
    if total_chunks_before_tourid_dedup > chunks_after_tourid_dedup:
        print(f"去重了 {total_chunks_before_tourid_dedup - chunks_after_tourid_dedup} 个文档块。")

    # --- 步骤 2.2: 为每个chunk生成唯一的ID (哈希值) ---
    processed_chunk_hashes_in_mem = set()
    all_chunk_ids_to_upsert = []  # 存储所有即将 upsert 到 ChromaDB 的哈希ID

    for chunk in chunks:
        calculated_hash = generate_chunk_hash(chunk)  # 调用辅助函数生成哈希

        # 确保 chunk.metadata 中存在 Chunk_ID_Hash 字段
        if EXTERNAL_METADATA_CHUNK_ID_COL not in chunk.metadata or chunk.metadata[
            EXTERNAL_METADATA_CHUNK_ID_COL] != calculated_hash:
            chunk.metadata[EXTERNAL_METADATA_CHUNK_ID_COL] = calculated_hash

        current_chunk_hash = chunk.metadata.get(EXTERNAL_METADATA_CHUNK_ID_COL)
        if current_chunk_hash is None:
            print(f"!!! 严重警告：文档块没有生成有效的 Chunk_ID_Hash，内容片段: {chunk.page_content[:100]}...")
            continue

        if current_chunk_hash in processed_chunk_hashes_in_mem:
            print(f"!!! 严重警告：在本次加载的文档中发现重复的 Chunk_ID_Hash: {current_chunk_hash}")
            print(f"  - 原始文件: {chunk.metadata.get('source', 'N/A')}, 内容片段: {chunk.page_content[:100]}...")
            raise ValueError(f"检测到重复的 Chunk_ID_Hash '{current_chunk_hash}'。请检查知识库文件是否有重复内容。")
        processed_chunk_hashes_in_mem.add(current_chunk_hash)
        all_chunk_ids_to_upsert.append(current_chunk_hash)

    print(f"--- [{datetime.datetime.now()}] 步骤 2.3: 手动清理元数据中的复杂类型 (列表/字典) 值... ---")
    for i, chunk in enumerate(chunks):
        cleaned_metadata = {}
        for key, value in chunk.metadata.items():
            if isinstance(value, (list, dict)):
                try:
                    cleaned_metadata[key] = json.dumps(value, ensure_ascii=False)
                except TypeError as e:
                    print(f"!!! 警告：Chunk {i} 的元数据 '{key}' 无法转换为JSON字符串：{e}。原始值: {value}")
                    cleaned_metadata[key] = str(value)
            else:
                cleaned_metadata[key] = value
        chunk.metadata = cleaned_metadata
    print(f"手动清理元数据完成。共处理 {len(chunks)} 个文档块。")

    print(f"--- [{datetime.datetime.now()}] 步骤 2.4: 导入外部 Excel 标注的元数据 (Category) ---")
    if os.path.exists(EXTERNAL_METADATA_EXCEL):
        try:
            external_df = pd.read_excel(EXTERNAL_METADATA_EXCEL)
            if EXTERNAL_METADATA_CHUNK_ID_COL in external_df.columns and EXTERNAL_METADATA_CATEGORY_COL in external_df.columns:
                external_metadata_map = {}
                for _, row in external_df.iterrows():
                    chunk_id = row[EXTERNAL_METADATA_CHUNK_ID_COL]
                    category = row[EXTERNAL_METADATA_CATEGORY_COL]
                    if pd.notna(chunk_id) and pd.notna(category) and str(category).strip() != '':
                        external_metadata_map[str(chunk_id)] = str(category).strip()

                if external_metadata_map:
                    updated_chunks_count = 0
                    for chunk in chunks:  # 遍历所有准备写入DB的chunk
                        current_chunk_id = chunk.metadata.get(EXTERNAL_METADATA_CHUNK_ID_COL)
                        if current_chunk_id and str(current_chunk_id) in external_metadata_map:
                            chunk.metadata[EXTERNAL_METADATA_CATEGORY_COL] = external_metadata_map[
                                str(current_chunk_id)]
                            updated_chunks_count += 1
                    print(f"成功导入了 {updated_chunks_count} 个文档块的 '{EXTERNAL_METADATA_CATEGORY_COL}' 元数据。")
                else:
                    print("外部元数据文件中没有找到有效的标注数据。请确保 Category 列有值。")
            else:
                print(
                    f"警告：外部元数据文件 '{EXTERNAL_METADATA_EXCEL}' 缺少列 '{EXTERNAL_METADATA_CHUNK_ID_COL}' 或 '{EXTERNAL_METADATA_CATEGORY_COL}'。跳过导入。")
        except Exception as e:
            print(f"!!! 警告：导入外部元数据文件 '{EXTERNAL_METADATA_EXCEL}' 失败：{e}")
    else:
        print(f"--- 未找到外部元数据文件: '{EXTERNAL_METADATA_EXCEL}'。跳过导入。---")

    # 打印前几个分块的效果及其元数据，用于验证
    print("\n--- 分块效果预览 (前5个分块) ---")
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- 分块 {i + 1} ---")
        truncated_content = chunk.page_content[:300].replace('\n', ' ')
        print(f"内容片段: {truncated_content}...")
        print(f"元数据: {chunk.metadata}")
        print("-" * 20)
    print("-----------------------------\n")


    print(f"--- [{datetime.datetime.now()}] 步骤 4 & 5: 存储到 ChromaDB (增量更新或全量重建) ---")

    # 获取所有即将 upsert 到 ChromaDB 的 chunk 的 ID
    all_chunk_ids_to_upsert = [chunk.metadata[EXTERNAL_METADATA_CHUNK_ID_COL] for chunk in chunks if
                               EXTERNAL_METADATA_CHUNK_ID_COL in chunk.metadata]
    if not all_chunk_ids_to_upsert:
        print("!!! 错误：没有有效的文档块哈希ID用于存储到 ChromaDB。")
        return

    # 这里不再重复判断 incremental_update，因为 db_instance 已经根据前面的逻辑初始化好了。

    if not incremental_update:
        # 全量重建：db_instance 已经在前面创建了一个新的空数据库
        print(f"--- [{datetime.datetime.now()}] 5: 存储到 ChromaDB (全量重建) ---")

        if chunks:
            added_ids = db_instance.add_documents(chunks, ids=all_chunk_ids_to_upsert)
            print(f"成功将 {len(added_ids)} 个文档块（全量重建）存储到 ChromaDB，存储路径: '{CHROMA_PATH}'。")
        else:
            print("没有文档块需要全量重建。")

    else:
        # 增量更新：更智能地处理新增、删除和修改
        print(f"--- [{datetime.datetime.now()}] 增量更新 ChromaDB ---")

        # 获取现有 ChromaDB 中所有 chunk 的哈希ID和内部ID
        existing_chroma_data = db_instance.get(include=[ 'metadatas'])
        existing_hash_to_chroma_id_map = {
            m.get(EXTERNAL_METADATA_CHUNK_ID_COL): cid
            for cid, m in zip(existing_chroma_data['ids'], existing_chroma_data['metadatas'])
            if m.get(EXTERNAL_METADATA_CHUNK_ID_COL)
        }
        existing_hashes_in_db = set(existing_hash_to_chroma_id_map.keys())
        print(f"现有 ChromaDB 中包含 {len(existing_hashes_in_db)} 个唯一文档块。")

        # 找出需要删除的 Chunk_ID_Hash (在DB中存在，但在本次加载中不再存在)

        # 从 new_document_status (本次加载成功的文件) 中收集所有 chunk_hashes
        hashes_from_processed_files = set()
        for file_id, file_info in new_document_status.items():
            if file_info and 'chunk_hashes' in file_info:
                hashes_from_processed_files.update(file_info['chunk_hashes'])

        ids_to_delete_from_chroma = []
        for existing_hash in existing_hashes_in_db:
            # 如果某个哈希在DB中存在，但不在当前处理的文件中（即被删除或修改了），则需要从DB删除
            if existing_hash not in hashes_from_processed_files:
                chroma_internal_id = existing_hash_to_chroma_id_map.get(existing_hash)
                if chroma_internal_id:
                    ids_to_delete_from_chroma.append(chroma_internal_id)

        if ids_to_delete_from_chroma:
            print(f"--- [{datetime.datetime.now()}] 正在删除 {len(ids_to_delete_from_chroma)} 个旧文档块... ---")
            db_instance.delete(ids=ids_to_delete_from_chroma)
            print(f"成功从 ChromaDB 删除了 {len(ids_to_delete_from_chroma)} 个文档块。")
        else:
            print("没有旧文档块需要删除。")

        # 执行添加/更新操作 (upsert)
        if chunks:
            print(f"--- [{datetime.datetime.now()}] 正在增量更新/添加 {len(chunks)} 个文档块... ---")
            added_or_updated_ids = db_instance.add_documents(chunks, ids=all_chunk_ids_to_upsert)
            print(f"成功增量更新/添加了 {len(added_or_updated_ids)} 个文档块到 ChromaDB，存储路径: '{CHROMA_PATH}'。")
        else:
            print("没有文档块需要增量更新或添加。")

    print(f"--- [{datetime.datetime.now()}] 索引创建完成！ ---")

    # --- 保存最终的文档状态 (无论增量还是全量) ---
    final_document_status_to_save = {}
    # 首先从 new_document_status 复制，它包含了未修改文件和已处理文件（更新了mtime）的最新信息
    for file_id, file_info in new_document_status.items():
        if file_info:
            final_document_status_to_save[file_id] = file_info

    # 然后，遍历所有的 chunk，更新每个文件的 chunk_hashes 列表
    # 这一步是为了确保 new_document_status 中每个文件的 'chunk_hashes' 字段是当前实际被索引的 chunk 列表
    # 如果文件是新文件，或者文件内容被修改，其chunk_hashes需要重新构建
    file_hashes_map = defaultdict(list)
    for chunk in chunks:
        file_path = chunk.metadata.get("source")
        if file_path:
            # 使用 os.path.relpath 确保正确处理相对路径，避免 Path().relative_to() 在不同盘符下的问题
            # 或者确保 KNOWLEDGE_BASE_DIR 和 file_path 在同一根路径下
            # 这里保持 Path().relative_to(KNOWLEDGE_BASE_DIR) 的逻辑，假设都在同一目录下
            try:
                relative_file_id = str(Path(file_path).relative_to(KNOWLEDGE_BASE_DIR))
            except ValueError:
                print(f"警告：文件路径 '{file_path}' 不在知识库目录 '{KNOWLEDGE_BASE_DIR}' 下。无法记录其状态。")
                continue

            current_chunk_hash = chunk.metadata.get(EXTERNAL_METADATA_CHUNK_ID_COL)
            if current_chunk_hash:
                file_hashes_map[relative_file_id].append(current_chunk_hash)

    for file_id, hashes in file_hashes_map.items():
        if file_id not in final_document_status_to_save:
            # 这是处理了但 new_document_status 中没有条目的情况（例如全新的文件）
            final_document_status_to_save[file_id] = {
                'last_mtime': os.path.getmtime(os.path.join(KNOWLEDGE_BASE_DIR, file_id)),
                'chunk_hashes': [] # 重新初始化
            }
        # 更新或设置 chunk_hashes 列表
        final_document_status_to_save[file_id]['chunk_hashes'] = hashes

    # 移除在本次运行中，文件不再存在于 KNOWLEDGE_BASE_DIR 但仍然在 old_document_status 中的记录
    all_current_files = set()
    for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            relative_file_id = str(Path(file_path).relative_to(KNOWLEDGE_BASE_DIR))
            all_current_files.add(relative_file_id)

    # 从最终状态中删除不再存在的文件
    keys_to_delete = [file_id for file_id in final_document_status_to_save.keys() if file_id not in all_current_files]
    for file_id in keys_to_delete:
        del final_document_status_to_save[file_id]
        print(f"已从状态文件中移除不存在的文件记录: {file_id}")


    save_document_status(final_document_status_to_save)
    print(f"--- [{datetime.datetime.now()}] 文档处理状态已保存到 '{DOCUMENT_STATUS_FILE}'。 ---")


if __name__ == "__main__":
    create_database(incremental_update=INCREMENTAL_UPDATE_DEFAULT)  # <-- 使用顶部配置变量