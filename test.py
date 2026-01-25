# =======================================================
# 步骤 3: 执行数据库存入/更新操作 (Upsert)
# =======================================================
if chunks:
    print(f"--- [{datetime.datetime.now()}] 正在执行分块存入 (共 {len(chunks)} 个)... ---")
    # 这里的 all_chunk_ids_to_upsert 是我们在合并循环中生成的唯一哈希列表
    added_or_updated_ids = db_instance.add_documents(chunks, ids=all_chunk_ids_to_upsert)
    print(f"✅ 成功同步 {len(added_or_updated_ids)} 个文档块到 ChromaDB")
else:
    print("💡 没有检测到变动，无需更新数据库。")

print(f"--- [{datetime.datetime.now()}] 索引同步完成！ ---")

# =======================================================
# 步骤 4: 同步并保存文档状态 (document_status.json)
# =======================================================
final_document_status_to_save = {}

# 1. 继承未修改文件的最新信息
for file_id, file_info in new_document_status.items():
    if file_info:
        final_document_status_to_save[file_id] = file_info

# 2. 更新本次处理过的文件的 chunk_hashes
# file_hashes_map 会统计本次运行中所有 chunk 属于哪个文件
file_hashes_map = defaultdict(list)
for chunk in chunks:
    file_path = chunk.metadata.get("source")
    if file_path:
        try:
            # 获取相对路径作为唯一标识键
            relative_file_id = str(Path(file_path).relative_to(KNOWLEDGE_BASE_DIR))
            chunk_hash = chunk.metadata.get(EXTERNAL_METADATA_CHUNK_ID_COL)
            if chunk_hash:
                file_hashes_map[relative_file_id].append(chunk_hash)
        except ValueError:
            continue

for file_id, hashes in file_hashes_map.items():
    full_path = os.path.join(KNOWLEDGE_BASE_DIR, file_id)
    if os.path.exists(full_path):
        final_document_status_to_save[file_id] = {
            'last_mtime': os.path.getmtime(full_path),
            'chunk_hashes': hashes
        }

# 3. 物理清理：移除已经从磁盘删除的文件记录
all_current_files = set()
for root, _, filenames in os.walk(KNOWLEDGE_BASE_DIR):
    for f in filenames:
        rel_path = str(Path(os.path.join(root, f)).relative_to(KNOWLEDGE_BASE_DIR))
        all_current_files.add(rel_path)

keys_to_delete = [fid for fid in final_document_status_to_save.keys() if fid not in all_current_files]
for fid in keys_to_delete:
    del final_document_status_to_save[fid]
    print(f"🗑️ 已移除不存在的文件状态记录: {fid}")

# 4. 写入 document_status.json
save_document_status(final_document_status_to_save)
print(f"💾 文档处理状态已保存。")

# =======================================================
# 步骤 5: 自动化版本审计 (生成 version_info.json)
# =======================================================
print("\n" + "=" * 30)
print("📝 正在生成本次构建的审计日志...")

# 获取 Git Commit ID (如果报错则返回未知)
import subprocess

try:
    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
except:
    git_hash = "No Git Repo"

# 处理手动备注：Config优先，否则屏幕输入
manual_note = getattr(config, 'CURRENT_VERSION_NOTE', None)
if not manual_note:
    manual_note = input("👉 请输入本次版本的备注说明（直接回车跳过）: ")
    if not manual_note: manual_note = "无备注"

audit_payload = {
    "version": config.ACTIVE_DB_VERSION,
    "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "embedding_model": config.EMBEDDING_MODEL,
    "git_commit": git_hash,
    "manual_notes": manual_note,
    "parameters": {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "header_splitting": "H1-H4"
    },
    "statistics": {
        "total_chunks_in_db": db_instance._collection.count(),
        "processed_files_count": len(final_document_status_to_save),
        "file_distribution": file_distribution  # 使用我们在循环中累加的字典
    }
}

# 自动保存到当前数据库所在目录
audit_path = os.path.join(CHROMA_PATH, "version_info.json")
with open(audit_path, 'w', encoding='utf-8') as f:
    json.dump(audit_payload, f, ensure_ascii=False, indent=4)

print(f"🏆 审计日志已存至: {audit_path}")
print("=" * 30 + "\n")

except Exception as e:
print(f"❌ 运行过程中发生严重错误: {e}")
import traceback

traceback.print_exc()

if __name__ == "__main__":
    # 直接运行创建数据库函数
    create_database(incremental_update=INCREMENTAL_UPDATE_DEFAULT)