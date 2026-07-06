# patch_qwen_deep_research_responses.py
# -*- coding: utf-8 -*-

"""
把 deep_research_literature_agent.py 改成支持真正的 qwen-deep-research + web_search + web_extractor。

它会做三件事：
1. 给 deep_research_literature_agent.py 增加 import requests。
2. 替换 DeepResearchLLM 类，使其支持 llm_api_mode=responses。
3. 修改/创建 llm_providers.json，把 dashscope.deep_research_api_mode 设置为 responses。

使用：
    python patch_qwen_deep_research_responses.py

然后运行：
    set DASHSCOPE_API_KEY=你的key
    python deep_research_literature_agent.py --max-results 2 --batch-size 1
"""

from __future__ import annotations

import json
import re
from pathlib import Path


TARGET = Path("deep_research_literature_agent.py")
BACKUP = Path("deep_research_literature_agent.py.bak_qwen_deep_research_responses")
PROVIDER_CONFIG = Path("llm_providers.json")

NEW_CLASS = 'class DeepResearchLLM:\n    def __init__(\n        self,\n        model: str = DEEP_RESEARCH_MODEL,\n        provider: Optional[str] = None,\n        provider_config_path: Path = DEFAULT_PROVIDER_CONFIG,\n        max_retries: Optional[int] = None,\n        timeout_seconds: Optional[float] = None,\n        use_response_format: Optional[bool] = None,\n    ) -> None:\n        self.provider = provider or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)\n        self.provider_config_path = provider_config_path\n        self.provider_cfg = load_llm_provider_config(\n            provider=self.provider,\n            config_path=self.provider_config_path,\n        )\n\n        self.model = model\n        self.max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "6"))\n        self.timeout_seconds = timeout_seconds or float(os.getenv("LLM_TIMEOUT_SECONDS", "600"))\n\n        self.api_mode = (\n            self.provider_cfg.get("deep_research_api_mode")\n            or self.provider_cfg.get("responses_api_mode")\n            or self.provider_cfg.get("llm_api_mode")\n            or "chat"\n        ).lower()\n\n        api_key_env = (\n            self.provider_cfg.get("deep_research_api_key_env")\n            or self.provider_cfg.get("responses_api_key_env")\n            or self.provider_cfg.get("chat_api_key_env")\n            or "OPENAI_API_KEY"\n        )\n        base_url = (\n            self.provider_cfg.get("deep_research_base_url")\n            or self.provider_cfg.get("responses_base_url")\n            or self.provider_cfg.get("chat_base_url")\n            or os.getenv("OPENAI_BASE_URL")\n        )\n\n        self.api_key_env = str(api_key_env)\n        self.api_key = os.getenv(self.api_key_env) or os.getenv("OPENAI_API_KEY")\n        self.base_url = str(base_url or "").rstrip("/")\n\n        if not self.api_key:\n            raise RuntimeError(\n                f"没有找到 API Key。请设置环境变量 {self.api_key_env}。\\n"\n                f"Windows CMD 示例：set {self.api_key_env}=你的key"\n            )\n\n        if not self.base_url:\n            raise RuntimeError(\n                "没有找到 base_url。请在 llm_providers.json 中设置 "\n                "deep_research_base_url/responses_base_url/chat_base_url。"\n            )\n\n        self.tools = self.provider_cfg.get("deep_research_tools", [])\n        self.poll_interval_seconds = float(os.getenv("RESPONSES_POLL_INTERVAL_SECONDS", "10"))\n        self.poll_timeout_seconds = float(os.getenv("RESPONSES_POLL_TIMEOUT_SECONDS", "1800"))\n\n        if use_response_format is None:\n            cfg_value = self.provider_cfg.get("use_response_format")\n            if cfg_value is None:\n                self.use_response_format = os.getenv("USE_RESPONSE_FORMAT", "0") == "1"\n            else:\n                self.use_response_format = bool(cfg_value)\n        else:\n            self.use_response_format = bool(use_response_format)\n\n        # chat 模式使用 OpenAI SDK；responses 模式使用 requests 直连 /responses。\n        self.client = None\n        if self.api_mode == "chat":\n            self.client = OpenAI(\n                api_key=self.api_key,\n                base_url=self.base_url,\n                timeout=self.timeout_seconds,\n                max_retries=0,\n            )\n\n    def _is_retryable_error(self, error: Exception) -> bool:\n        name = error.__class__.__name__\n        message = str(error).lower()\n\n        retryable_names = {\n            "APITimeoutError",\n            "APIConnectionError",\n            "RateLimitError",\n            "InternalServerError",\n            "ConnectTimeout",\n            "ReadTimeout",\n            "TimeoutException",\n            "RemoteProtocolError",\n            "ConnectionError",\n            "Timeout",\n        }\n\n        retryable_keywords = [\n            "timed out",\n            "timeout",\n            "connection",\n            "temporarily unavailable",\n            "rate limit",\n            "429",\n            "500",\n            "502",\n            "503",\n            "504",\n            "ssl",\n            "eof",\n        ]\n\n        return name in retryable_names or any(keyword in message for keyword in retryable_keywords)\n\n    def chat(self, agent: MarkdownAgent, user_prompt: str) -> str:\n        if self.api_mode == "responses":\n            return self._responses_chat(agent, user_prompt)\n        return self._chat_completions(agent, user_prompt)\n\n    def _chat_completions(self, agent: MarkdownAgent, user_prompt: str) -> str:\n        kwargs = {\n            "model": self.model,\n            "messages": [\n                {"role": "system", "content": agent.system},\n                {"role": "user", "content": user_prompt},\n            ],\n            "temperature": agent.temperature,\n        }\n\n        if (\n            self.use_response_format\n            and agent.response_format\n            and agent.response_format.lower() == "json_object"\n        ):\n            kwargs["response_format"] = {"type": "json_object"}\n\n        last_error = None\n\n        for attempt in range(1, self.max_retries + 1):\n            try:\n                print(\n                    f"       ↳ LLM 请求(chat): provider={self.provider}, "\n                    f"agent={agent.agent_id}, model={self.model}, "\n                    f"timeout={self.timeout_seconds}s, attempt={attempt}/{self.max_retries}"\n                )\n                response = self.client.chat.completions.create(**kwargs)\n                return response.choices[0].message.content or ""\n\n            except Exception as e:\n                last_error = e\n\n                if not self._is_retryable_error(e):\n                    raise\n\n                wait = min(5 * (2 ** (attempt - 1)), 90)\n                print(\n                    f"⚠️ LLM 网络/超时错误，{wait}s 后重试 "\n                    f"({attempt}/{self.max_retries}): {e.__class__.__name__}: {e}"\n                )\n                time.sleep(wait)\n\n        raise RuntimeError(f"LLM chat 调用多次重试后仍失败: {last_error}")\n\n    def _responses_url(self) -> str:\n        return self.base_url + "/responses"\n\n    def _response_get_url(self, response_id: str) -> str:\n        return self.base_url + f"/responses/{response_id}"\n\n    def _tool_specs(self) -> List[Dict[str, Any]]:\n        specs = []\n        for tool in self.tools or []:\n            if isinstance(tool, str):\n                specs.append({"type": tool})\n            elif isinstance(tool, dict):\n                specs.append(tool)\n        return specs\n\n    def _responses_payload(self, agent: MarkdownAgent, user_prompt: str) -> Dict[str, Any]:\n        payload: Dict[str, Any] = {\n            "model": self.model,\n            "input": [\n                {"role": "system", "content": agent.system},\n                {"role": "user", "content": user_prompt},\n            ],\n        }\n\n        tools = self._tool_specs()\n        if tools:\n            payload["tools"] = tools\n\n        # 某些 Responses 实现支持 temperature；如果你的平台报参数错误，可在 llm_providers.json 里设 include_temperature=false。\n        if self.provider_cfg.get("include_temperature", True):\n            payload["temperature"] = agent.temperature\n\n        # Deep Research 通常耗时较长。如果服务端支持 background，可配置启用。\n        if self.provider_cfg.get("background", False):\n            payload["background"] = True\n\n        return payload\n\n    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:\n        headers = {\n            "Authorization": f"Bearer {self.api_key}",\n            "Content-Type": "application/json",\n        }\n        response = requests.post(\n            url,\n            headers=headers,\n            json=payload,\n            timeout=self.timeout_seconds,\n        )\n\n        if response.status_code >= 400:\n            text = response.text[:3000]\n            raise RuntimeError(\n                f"Responses API 请求失败: HTTP {response.status_code}\\n"\n                f"URL: {url}\\n"\n                f"Response: {text}"\n            )\n\n        return response.json()\n\n    def _get_json(self, url: str) -> Dict[str, Any]:\n        headers = {\n            "Authorization": f"Bearer {self.api_key}",\n            "Content-Type": "application/json",\n        }\n        response = requests.get(\n            url,\n            headers=headers,\n            timeout=self.timeout_seconds,\n        )\n\n        if response.status_code >= 400:\n            text = response.text[:3000]\n            raise RuntimeError(\n                f"Responses API 轮询失败: HTTP {response.status_code}\\n"\n                f"URL: {url}\\n"\n                f"Response: {text}"\n            )\n\n        return response.json()\n\n    def _responses_chat(self, agent: MarkdownAgent, user_prompt: str) -> str:\n        payload = self._responses_payload(agent, user_prompt)\n        last_error = None\n\n        for attempt in range(1, self.max_retries + 1):\n            try:\n                print(\n                    f"       ↳ LLM 请求(responses): provider={self.provider}, "\n                    f"agent={agent.agent_id}, model={self.model}, "\n                    f"tools={self.tools}, timeout={self.timeout_seconds}s, "\n                    f"attempt={attempt}/{self.max_retries}"\n                )\n                data = self._post_json(self._responses_url(), payload)\n                data = self._wait_for_response_if_needed(data)\n                text = self._extract_response_text(data)\n                if not text.strip():\n                    raise RuntimeError(\n                        "Responses API 返回为空文本。原始返回前 3000 字符："\n                        + json.dumps(data, ensure_ascii=False)[:3000]\n                    )\n                return text\n\n            except Exception as e:\n                last_error = e\n\n                # 400/404 这类参数错误没必要重试。\n                if "HTTP 400" in str(e) or "HTTP 404" in str(e):\n                    raise\n\n                if not self._is_retryable_error(e):\n                    raise\n\n                wait = min(5 * (2 ** (attempt - 1)), 90)\n                print(\n                    f"⚠️ Responses API 网络/超时错误，{wait}s 后重试 "\n                    f"({attempt}/{self.max_retries}): {e.__class__.__name__}: {e}"\n                )\n                time.sleep(wait)\n\n        raise RuntimeError(f"Responses API 调用多次重试后仍失败: {last_error}")\n\n    def _wait_for_response_if_needed(self, data: Dict[str, Any]) -> Dict[str, Any]:\n        status = str(data.get("status", "")).lower()\n        response_id = data.get("id")\n\n        if status not in {"queued", "in_progress", "running", "pending"}:\n            return data\n\n        if not response_id:\n            return data\n\n        start = time.time()\n        while time.time() - start < self.poll_timeout_seconds:\n            print(f"       ↳ Responses 任务状态: {status}，等待 {self.poll_interval_seconds}s 后轮询...")\n            time.sleep(self.poll_interval_seconds)\n            data = self._get_json(self._response_get_url(str(response_id)))\n            status = str(data.get("status", "")).lower()\n\n            if status in {"completed", "succeeded", "success", "done"}:\n                return data\n            if status in {"failed", "cancelled", "canceled", "expired"}:\n                raise RuntimeError(\n                    "Responses API 任务失败："\n                    + json.dumps(data, ensure_ascii=False)[:3000]\n                )\n\n        raise TimeoutError(\n            f"Responses API 轮询超时，response_id={response_id}, "\n            f"timeout={self.poll_timeout_seconds}s"\n        )\n\n    def _extract_response_text(self, data: Dict[str, Any]) -> str:\n        # OpenAI Responses 常见字段\n        if isinstance(data.get("output_text"), str) and data["output_text"].strip():\n            return data["output_text"]\n\n        texts: List[str] = []\n\n        def collect_from_content(content: Any) -> None:\n            if isinstance(content, str):\n                if content.strip():\n                    texts.append(content)\n                return\n\n            if isinstance(content, list):\n                for part in content:\n                    if isinstance(part, str):\n                        if part.strip():\n                            texts.append(part)\n                    elif isinstance(part, dict):\n                        if isinstance(part.get("text"), str):\n                            texts.append(part["text"])\n                        elif isinstance(part.get("content"), str):\n                            texts.append(part["content"])\n                return\n\n            if isinstance(content, dict):\n                if isinstance(content.get("text"), str):\n                    texts.append(content["text"])\n                elif isinstance(content.get("content"), str):\n                    texts.append(content["content"])\n\n        output = data.get("output")\n        if isinstance(output, list):\n            for item in output:\n                if isinstance(item, dict):\n                    collect_from_content(item.get("content"))\n                    if isinstance(item.get("text"), str):\n                        texts.append(item["text"])\n\n        choices = data.get("choices")\n        if isinstance(choices, list):\n            for choice in choices:\n                if not isinstance(choice, dict):\n                    continue\n                msg = choice.get("message") or {}\n                collect_from_content(msg.get("content"))\n                collect_from_content(choice.get("text"))\n\n        # 一些服务会放在 data/message/content\n        collect_from_content(data.get("content"))\n        collect_from_content(data.get("text"))\n\n        return "\\n".join(t for t in texts if str(t).strip()).strip()'


def find_class_region(text: str) -> tuple[int, int]:
    start_match = re.search(r"^class\s+DeepResearchLLM\s*:", text, flags=re.MULTILINE)
    if not start_match:
        raise RuntimeError("没有找到 class DeepResearchLLM。")

    start = start_match.start()
    end_match = re.search(r"^def\s+build_records_context\s*\(", text[start:], flags=re.MULTILINE)
    if end_match:
        return start, start + end_match.start()

    raise RuntimeError("找到了 DeepResearchLLM，但没有找到后面的 build_records_context。")


def ensure_imports(text: str) -> str:
    if re.search(r"^import\s+requests\s*$", text, flags=re.MULTILINE):
        return text

    # 插到 import urllib.request 后面；找不到则插到 import time 后面。
    if "import urllib.request" in text:
        return text.replace("import urllib.request", "import urllib.request\nimport requests", 1)

    if "import time" in text:
        return text.replace("import time", "import time\nimport requests", 1)

    return "import requests\n" + text


def patch_python_file() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"没有找到 {TARGET}，请把补丁放到项目根目录运行。")

    text = TARGET.read_text(encoding="utf-8", errors="ignore")

    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")

    text = ensure_imports(text)
    start, end = find_class_region(text)
    new_text = text[:start] + NEW_CLASS + "\n\n\n" + text[end:].lstrip()

    TARGET.write_text(new_text, encoding="utf-8")


def patch_provider_config() -> None:
    if PROVIDER_CONFIG.exists():
        data = json.loads(PROVIDER_CONFIG.read_text(encoding="utf-8"))
    else:
        data = {}

    dashscope = data.setdefault("dashscope", {})
    dashscope.update({
        "provider": "dashscope",
        "display_name": "Alibaba Bailian / DashScope",
        "chat_api_key_env": "DASHSCOPE_API_KEY",
        "chat_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "deepseek-v4-pro",
        "responses_api_key_env": "DASHSCOPE_API_KEY",
        "responses_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "responses_model": "deepseek-v4-pro",
        "deep_research_api_key_env": "DASHSCOPE_API_KEY",
        "deep_research_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "deep_research_provider": "dashscope",
        "deep_research_model": "qwen-deep-research",
        "deep_research_api_mode": "responses",
        "deep_research_tools": ["web_search", "web_extractor"],
        "llm_api_mode": "responses",
        "use_response_format": False,
        "include_temperature": True,
        "background": False
    })

    PROVIDER_CONFIG.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    patch_python_file()
    patch_provider_config()

    print("✅ 已接入 qwen-deep-research Responses API 分支")
    print(f"   已备份: {BACKUP}")
    print(f"   已修改: {TARGET}")
    print(f"   已更新: {PROVIDER_CONFIG}")
    print()
    print("建议先小批量测试：")
    print("   set DASHSCOPE_API_KEY=你的key")
    print("   set LLM_TIMEOUT_SECONDS=600")
    print("   set RESPONSES_POLL_TIMEOUT_SECONDS=1800")
    print("   python deep_research_literature_agent.py --max-results 2 --batch-size 1")
    print()
    print("如果服务端报参数错误 include_temperature，可在 llm_providers.json 里把 include_temperature 改成 false。")
    print("如果服务端要求后台模式，可在 llm_providers.json 里把 background 改成 true。")


if __name__ == "__main__":
    main()
