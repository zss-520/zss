# patch_dotenv_support.py
# -*- coding: utf-8 -*-

"""
给 deep_research_literature_agent.py 添加 .env 自动读取支持。

解决问题：
    .env 里面写了 DASHSCOPE_API_KEY，但是 os.getenv("DASHSCOPE_API_KEY") 读取不到。

使用方法：
    pip install python-dotenv
    python patch_dotenv_support.py

补丁会自动备份：
    deep_research_literature_agent.py.bak_dotenv
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("deep_research_literature_agent.py")
BACKUP = Path("deep_research_literature_agent.py.bak_dotenv")

DOTENV_BLOCK = '# ===== Load .env early =====\n# 必须放在所有 os.getenv(...) 调用之前。\ntry:\n    from dotenv import load_dotenv\n\n    _PROJECT_ROOT = Path(__file__).resolve().parent\n    _ENV_PATH = _PROJECT_ROOT / ".env"\n\n    if _ENV_PATH.exists():\n        load_dotenv(_ENV_PATH, override=False)\n    else:\n        load_dotenv(override=False)\nexcept Exception:\n    # 没安装 python-dotenv 时，不影响系统环境变量读取。\n    pass\n# ===== End load .env =====\n\n'


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"没有找到 {TARGET}。请把本补丁放在项目根目录，也就是和 deep_research_literature_agent.py 同级。"
        )

    text = TARGET.read_text(encoding="utf-8", errors="ignore")

    if "===== Load .env early =====" in text:
        print("✅ deep_research_literature_agent.py 已经包含 .env 读取逻辑，无需重复修改。")
        return

    if "from pathlib import Path" not in text:
        if "import os" in text:
            text = text.replace("import os", "import os\nfrom pathlib import Path", 1)
        else:
            text = "import os\nfrom pathlib import Path\n" + text

    insert_after = "from pathlib import Path"
    idx = text.find(insert_after)
    if idx == -1:
        raise RuntimeError("没有找到 from pathlib import Path，无法自动插入。")

    insert_pos = idx + len(insert_after)
    new_text = text[:insert_pos] + "\n\n" + DOTENV_BLOCK + text[insert_pos:].lstrip("\n")

    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")

    TARGET.write_text(new_text, encoding="utf-8")

    print("✅ 已添加 .env 自动读取支持")
    print(f"   已备份: {BACKUP}")
    print(f"   已修改: {TARGET}")
    print()
    print("请确认 .env 文件在项目根目录，例如：")
    print("   E:\\agentbenmark_amp\\zss\\.env")
    print()
    print(".env 内容格式：")
    print("   DASHSCOPE_API_KEY=sk-你的真实key")
    print()
    print("然后运行：")
    print("   python deep_research_literature_agent.py --max-results 2 --batch-size 1")


if __name__ == "__main__":
    main()
