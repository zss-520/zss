# fix_json_dumps_sort_keys.py
# -*- coding: utf-8 -*-
"""
修复 deep_research_literature_agent.py 中：
TypeError: json_dumps() got an unexpected keyword argument 'sort_keys'

用法：
    python fix_json_dumps_sort_keys.py

它会备份：
    deep_research_literature_agent.py.bak_json_dumps_fix
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("deep_research_literature_agent.py")
BACKUP = Path("deep_research_literature_agent.py.bak_json_dumps_fix")

NEW_FUNC = '''def json_dumps(obj: Any, **kwargs) -> str:
    """统一 JSON 序列化函数，兼容 sort_keys 等 json.dumps 参数。"""
    params = {
        "ensure_ascii": False,
        "indent": 2,
    }
    params.update(kwargs)
    return json.dumps(obj, **params)
'''


def patch_function(text: str) -> tuple[str, bool]:
    # 匹配常见的 json_dumps 函数定义，到下一个顶格 def/class 之前
    pattern = re.compile(
        r"^def\s+json_dumps\s*\([^\)]*\)\s*(?:->\s*str\s*)?:\s*\n"
        r"(?:^[ \t]+.*\n)+",
        flags=re.MULTILINE,
    )

    match = pattern.search(text)
    if not match:
        return text, False

    # 尽量只替换函数体，不吞掉下一个顶格 def/class；如果 regex 吞多了，下面兜底不做复杂处理
    old = match.group(0)
    # 如果意外包含两个函数定义，截断到第二个顶格 def/class
    lines = old.splitlines(keepends=True)
    cut = len(lines)
    for i, line in enumerate(lines[1:], start=1):
        if re.match(r"^(def|class)\s+", line):
            cut = i
            break
    old_func = "".join(lines[:cut])
    rest = "".join(lines[cut:])
    return text.replace(old_func, NEW_FUNC + "\n", 1), True


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError("没有找到 deep_research_literature_agent.py，请把本脚本放到项目根目录运行。")

    text = TARGET.read_text(encoding="utf-8", errors="ignore")

    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")

    new_text, changed = patch_function(text)

    if not changed:
        # 如果没找到函数，至少把出错调用改成原生 json.dumps
        new_text = text.replace(
            "stable_hash(json_dumps(item, sort_keys=True))",
            "stable_hash(json.dumps(item, ensure_ascii=False, sort_keys=True))",
        )
        changed = new_text != text

    if not changed:
        raise RuntimeError(
            "没有找到可自动修复的位置。请手动把 json_dumps 函数改成支持 **kwargs，"
            "或者把 json_dumps(item, sort_keys=True) 改成 json.dumps(item, ensure_ascii=False, sort_keys=True)。"
        )

    TARGET.write_text(new_text, encoding="utf-8")
    print("✅ 修复完成")
    print(f"   已备份: {BACKUP}")
    print(f"   已修改: {TARGET}")
    print()
    print("现在重新运行：")
    print("   python deep_research_literature_agent.py --reprocess --max-results 10 --batch-size 5")
    print()
    print("如果不想重新抓全文，可以直接运行原命令；已处理过的 PMID 会按 index 跳过。")


if __name__ == "__main__":
    main()
