import os
from pathlib import Path
import pandas as pd
import re

def natural_sort_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', path.name)]

def merge_md_by_page_order(root_dir, output_file):
    """
    遍历每个子目录，将所有 page_x.md 按数字顺序合并
    每个子目录 → 表格中的一行
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"目录不存在: {root_path}")

    data = []

    for sub_dir in root_path.iterdir():
        if not sub_dir.is_dir():
            continue

        # 匹配 page_x.md 格式的文件
        md_files = list(sub_dir.glob("*_page_*.md"))
        if not md_files:
            print(f"⚠️  {sub_dir.name} 中没有 page_x.md 文件，跳过...")
            continue
        sorted_files = sorted(md_files, key=natural_sort_key)
        print(f"📄 处理子目录: {sub_dir.name} ({len(sorted_files)} 页)")
        combined_content = ""
        for md_file in sorted_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                # 添加页码注释（可选，便于追溯）
                page_match = re.search(r'_page_(\d+)', md_file.name)
                page_num = page_match.group(1) if page_match else "unknown"
                combined_content += f"{content}"
            except Exception as e:
                print(f"❌ 读取 {md_file.name} 失败: {e}")

        data.append({
            'name': sub_dir.name,
            'content': combined_content.strip()
        })

    # 输出表格
    if data:
        df = pd.DataFrame(data)
        df.to_excel(output_file, index=False)
        print(f"✅ 合并完成，结果已保存至: {output_file}")
    else:
        print("❌ 未找到任何符合条件的 .md 文件。")

# 使用示例
if __name__ == "__main__":
    root_directory = "./user_data/tmp_data/output"  # pdf 解析结果的文件路径
    output_excel = "./user_data/tmp_data/md.xlsx"
    merge_md_by_page_order(root_directory, output_excel)
