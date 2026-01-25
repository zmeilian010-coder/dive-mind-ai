import re
# --- Markdown 文件处理 (修正版) ---
if file_extension == ".md":
    file_stem = Path(file_path).stem # 获取文件名（不带后缀）作为 Category

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
            ("####", "Header4"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        # 图片现在跟着文字被分到了对应的 md_sections 里
        md_sections = header_splitter.split_text(markdown_content)

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )

        for section in md_sections:
            # 第一步：先提取 (从原始的 section.page_content 里抓 URL)
            section_images = re.findall(r'!\[.*?\]\((.*?)\)', section.page_content)

            # 第二步：后删除 (抓完之后，再把原文里的图片语法抹掉)
            section.page_content = re.sub(r'!\[.*?\]\((.*?)\)', '', section.page_content)

            # 准备元数据
            new_meta = common_metadata.copy()
            new_meta["version"] = MARKDOWN_DOC_VERSION
            new_meta["category"] = file_stem
            new_meta.update(section.metadata)

            # 自动识别习题
            header_values = [str(v) for v in section.metadata.values()]
            new_meta["is_quiz"] = any("习题与答案" in val for val in header_values)

            # 注入提取到的图片
            if section_images:
                new_meta["images"] = ",".join(section_images)
            else:
                new_meta["images"] = None

            # 规范化并执行二次切分
            section.metadata = {k: v for k, v in new_meta.items() if _normalize_metadata_value(k, v) is not None}
            file_chunks.extend(recursive_splitter.split_documents([section]))

        print(f"✅ 成功解析 Markdown: {file_stem}")

    except Exception as e:
        print(f"!!! 错误：处理 Markdown 文件 '{file_path}' 失败：{e}"