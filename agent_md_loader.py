# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping


_INCLUDE_RE = re.compile(r"\{\{include:([^{}]+)\}\}")
_VARIABLE_RE = re.compile(r"\{\{var:([A-Za-z_][A-Za-z0-9_]*)\}\}")


class AgentMDLoader:
    """Load UTF-8 Markdown prompts from one explicitly scoped directory."""

    def __init__(self, base_dir: str | Path, *, shared_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        default_shared = self.base_dir.parent / "shared"
        self.shared_dir = Path(shared_dir).expanduser().resolve() if shared_dir else default_shared.resolve()

    def path_for(self, name: str) -> Path:
        relative = Path(name)
        if relative.suffix == "":
            relative = relative.with_suffix(".md")
        path = (self.base_dir / relative).resolve()
        if path != self.base_dir and self.base_dir not in path.parents:
            raise ValueError(f"Agent prompt path escapes base directory: {name}")
        return path

    def load(self, name: str) -> str:
        path = self.path_for(name)
        if not path.is_file():
            raise FileNotFoundError(f"Agent prompt not found: {path}")
        return path.read_text(encoding="utf-8", errors="strict")

    def _include_path(self, reference: str) -> Path:
        reference = reference.strip()
        if reference.startswith("shared/"):
            relative = Path(reference.removeprefix("shared/"))
            root = self.shared_dir
        else:
            relative = Path(reference)
            root = self.base_dir
        if relative.suffix == "":
            relative = relative.with_suffix(".md")
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Agent prompt include escapes allowed directory: {reference}")
        if not path.is_file():
            raise FileNotFoundError(f"Included Agent prompt not found: {path}")
        return path

    def load_composed(self, name: str) -> str:
        """Load a prompt and recursively expand ``{{include:...}}`` directives."""

        def expand(text: str, stack: tuple[Path, ...]) -> str:
            def replace(match: re.Match[str]) -> str:
                path = self._include_path(match.group(1))
                if path in stack:
                    chain = " -> ".join(str(item) for item in (*stack, path))
                    raise ValueError(f"Circular Agent prompt include: {chain}")
                included = path.read_text(encoding="utf-8", errors="strict")
                return expand(included, (*stack, path)).strip()

            return _INCLUDE_RE.sub(replace, text)

        source = self.path_for(name)
        return expand(self.load(name), (source,))

    def render(
        self,
        name: str,
        variables: Mapping[str, object] | None = None,
        *,
        composed: bool = False,
    ) -> str:
        """Render explicit ``{{var:name}}`` tokens without interpreting JSON braces."""
        text = self.load_composed(name) if composed else self.load(name)
        values = {str(key): str(value) for key, value in (variables or {}).items()}

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise KeyError(f"Missing Agent prompt variable: {key}")
            return values[key]

        rendered = _VARIABLE_RE.sub(replace, text)
        unresolved = _VARIABLE_RE.search(rendered)
        if unresolved:
            raise KeyError(f"Unresolved Agent prompt variable: {unresolved.group(1)}")
        return rendered
