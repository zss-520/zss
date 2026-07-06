# -*- coding: utf-8 -*-
from pathlib import Path

class AgentMDLoader:
    """Simple Markdown agent prompt loader."""
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
    def load(self, name: str) -> str:
        path = self.base_dir / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Agent prompt not found: {path}")
        return path.read_text(encoding="utf-8")
