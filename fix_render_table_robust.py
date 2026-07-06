# fix_render_table_robust.py
# -*- coding: utf-8 -*-

"""
修复 deep_research_literature_agent.py 的 render_table() 函数。

解决错误：
    AttributeError: 'str' object has no attribute 'get'

原因：
    DeepSeek 有时会把 benchmark_implications / open_questions 等字段返回为字符串列表：
        ["Accuracy alone is not enough", "Need external validation"]
    但旧版 render_table 假设每一项都是 dict：
        it.get(c, '')
    所以遇到 str 就报错。

使用：
    python fix_render_table_robust.py

然后重新运行：
    python deep_research_literature_agent.py --reprocess --max-results 10 --batch-size 5
"""

from __future__ import annotations

import re
from pathlib import Path


TARGET = Path("deep_research_literature_agent.py")
BACKUP = Path("deep_research_literature_agent.py.bak_render_table_robust")


NEW_FUNC = r'''def render_table(items: Any, cols: List[str]) -> str:
    """
    将 list[dict] 渲染成 Markdown 表格。

    兼容 LLM 偶尔返回的不规范结构：
    - list[str]
    - list[list]
    - dict
    - None

    这样即使 DeepSeek 把 benchmark_implications / open_questions 返回成字符串列表，
    也不会因为 str 没有 .get() 而中断整个流程。
    """
    if not items:
        return "_无_\n"

    if isinstance(items, dict):
        items = [items]
    elif not isinstance(items, list):
        items = [items]

    normalized_items = []
    first_col = cols[0] if cols else "item"

    for item in items:
        if isinstance(item, dict):
            normalized_items.append(item)
        elif isinstance(item, list):
            normalized_items.append({first_col: "; ".join(str(x) for x in item)})
        else:
            normalized_items.append({first_col: str(item)})

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]

    for item in normalized_items:
        row = []
        for col in cols:
            value = item.get(col, "")
            if isinstance(value, (dict, list)):
                value = json_dumps(value)
            value = str(value).replace("|", "\\|").replace("\n", " ").strip()
            row.append(value[:800])
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows) + "\n"'''


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"没有找到 {TARGET}。请把本补丁放到项目根目录，也就是和 deep_research_literature_agent.py 同级。"
        )

    text = TARGET.read_text(encoding="utf-8", errors="ignore")

    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")

    pattern = re.compile(
        r"def\s+render_table\s*\([^)]*\)\s*->\s*str\s*:\s*\n(?:(?:[ \t]+.*|[ \t]*)\n)+?(?=^def\s+render_memory_md\s*\()",
        flags=re.MULTILINE,
    )

    match = pattern.search(text)
    if not match:
        raise RuntimeError(
            "没有找到 render_table 函数或 render_memory_md 边界。\n"
            "请手动搜索 def render_table，并用 README 里的函数替换。"
        )

    new_text = text[: match.start()] + NEW_FUNC + "\n\n" + text[match.end():]
    TARGET.write_text(new_text, encoding="utf-8")

    print("✅ render_table 已修复为鲁棒版本")
    print(f"   已备份: {BACKUP}")
    print(f"   已修改: {TARGET}")
    print()
    print("现在重新运行：")
    print("   python deep_research_literature_agent.py --reprocess --max-results 10 --batch-size 5")


if __name__ == "__main__":
    main()
