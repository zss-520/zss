# fix_json_dumps_full_compat.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("deep_research_literature_agent.py")
BACKUP = Path("deep_research_literature_agent.py.bak_json_dumps_full_compat")

NEW_FUNC = '''def json_dumps(obj: Any, indent: int = 2, **kwargs: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        indent=indent,
        **kwargs,
    )'''


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"没有找到 {TARGET}。请把本补丁放到项目根目录，也就是和 deep_research_literature_agent.py 同级。"
        )

    text = TARGET.read_text(encoding="utf-8", errors="ignore")

    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")

    pattern = re.compile(
        r"def\s+json_dumps\s*\([^)]*\)\s*->\s*str\s*:\s*\n(?:[ \t]+.*\n)+",
        flags=re.MULTILINE,
    )

    match = pattern.search(text)
    if not match:
        raise RuntimeError("没有找到 json_dumps 函数，请手动搜索 def json_dumps。")

    new_text = text[: match.start()] + NEW_FUNC + "\n\n" + text[match.end():]
    TARGET.write_text(new_text, encoding="utf-8")

    print("✅ json_dumps 已修复为完全兼容版本")
    print(f"   已备份: {BACKUP}")
    print(f"   已修改: {TARGET}")
    print()
    print("现在重新运行：")
    print("   python deep_research_literature_agent.py --reprocess --max-results 10 --batch-size 5")


if __name__ == "__main__":
    main()
