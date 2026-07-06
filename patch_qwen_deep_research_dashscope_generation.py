# patch_qwen_deep_research_dashscope_generation.py
# -*- coding: utf-8 -*-

"""
把 deep_research_literature_agent.py 改成使用真正的 DashScope SDK 调用：

    dashscope.Generation.call(
        model="qwen-deep-research",
        messages=messages,
        stream=True
    )

这个版本不走：
- OpenAI compatible chat.completions
- OpenAI compatible /responses

使用：
    python patch_qwen_deep_research_dashscope_generation.py

然后：
    pip install -U dashscope
    set DASHSCOPE_API_KEY=你的key
    python deep_research_literature_agent.py --max-results 2 --batch-size 1
"""

from __future__ import annotations

import json
import re
from pathlib import Path


TARGET = Path("deep_research_literature_agent.py")
BACKUP = Path("deep_research_literature_agent.py.bak_dashscope_generation")
PROVIDER_CONFIG = Path("llm_providers.json")

NEW_CLASS = 'class DeepResearchLLM:\n    def __init__(\n        self,\n        model: str = DEEP_RESEARCH_MODEL,\n        provider: Optional[str] = None,\n        provider_config_path: Path = DEFAULT_PROVIDER_CONFIG,\n        max_retries: Optional[int] = None,\n        timeout_seconds: Optional[float] = None,\n        use_response_format: Optional[bool] = None,\n    ) -> None:\n        self.provider = provider or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)\n        self.provider_config_path = provider_config_path\n        self.provider_cfg = load_llm_provider_config(\n            provider=self.provider,\n            config_path=self.provider_config_path,\n        )\n\n        self.model = model\n        self.max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "6"))\n        self.timeout_seconds = timeout_seconds or float(os.getenv("LLM_TIMEOUT_SECONDS", "600"))\n\n        self.api_mode = (\n            self.provider_cfg.get("deep_research_api_mode")\n            or self.provider_cfg.get("llm_api_mode")\n            or "chat"\n        ).lower()\n\n        self.api_key_env = str(\n            self.provider_cfg.get("deep_research_api_key_env")\n            or self.provider_cfg.get("responses_api_key_env")\n            or self.provider_cfg.get("chat_api_key_env")\n            or "OPENAI_API_KEY"\n        )\n        self.api_key = os.getenv(self.api_key_env) or os.getenv("OPENAI_API_KEY")\n\n        if not self.api_key:\n            raise RuntimeError(\n                f"没有找到 API Key。请设置环境变量 {self.api_key_env}。\\n"\n                f"Windows CMD 示例：set {self.api_key_env}=你的key"\n            )\n\n        self.base_url = str(\n            self.provider_cfg.get("deep_research_base_url")\n            or self.provider_cfg.get("responses_base_url")\n            or self.provider_cfg.get("chat_base_url")\n            or os.getenv("OPENAI_BASE_URL")\n            or ""\n        ).rstrip("/")\n\n        if use_response_format is None:\n            cfg_value = self.provider_cfg.get("use_response_format")\n            if cfg_value is None:\n                self.use_response_format = os.getenv("USE_RESPONSE_FORMAT", "0") == "1"\n            else:\n                self.use_response_format = bool(cfg_value)\n        else:\n            self.use_response_format = bool(use_response_format)\n\n        self.client = None\n\n        # OpenAI-compatible 普通 chat 模式。\n        if self.api_mode == "chat":\n            if not self.base_url:\n                raise RuntimeError("chat 模式需要 base_url，请检查 llm_providers.json。")\n            self.client = OpenAI(\n                api_key=self.api_key,\n                base_url=self.base_url,\n                timeout=self.timeout_seconds,\n                max_retries=0,\n            )\n\n        # qwen-deep-research 必须走 dashscope.Generation.call。\n        if self.api_mode == "dashscope_generation":\n            try:\n                import dashscope as _dashscope\n            except Exception as e:\n                raise RuntimeError(\n                    "缺少 dashscope 包。请先运行：pip install -U dashscope\\n"\n                    f"原始错误: {e}"\n                )\n            self.dashscope = _dashscope\n\n    def _is_retryable_error(self, error: Exception) -> bool:\n        name = error.__class__.__name__\n        message = str(error).lower()\n\n        retryable_names = {\n            "APITimeoutError",\n            "APIConnectionError",\n            "RateLimitError",\n            "InternalServerError",\n            "ConnectTimeout",\n            "ReadTimeout",\n            "TimeoutException",\n            "RemoteProtocolError",\n            "ConnectionError",\n            "Timeout",\n        }\n\n        retryable_keywords = [\n            "timed out",\n            "timeout",\n            "connection",\n            "temporarily unavailable",\n            "rate limit",\n            "429",\n            "500",\n            "502",\n            "503",\n            "504",\n            "ssl",\n            "eof",\n        ]\n\n        return name in retryable_names or any(keyword in message for keyword in retryable_keywords)\n\n    def chat(self, agent: MarkdownAgent, user_prompt: str) -> str:\n        if self.api_mode == "dashscope_generation":\n            return self._dashscope_generation_chat(agent, user_prompt)\n        return self._chat_completions(agent, user_prompt)\n\n    def _chat_completions(self, agent: MarkdownAgent, user_prompt: str) -> str:\n        kwargs = {\n            "model": self.model,\n            "messages": [\n                {"role": "system", "content": agent.system},\n                {"role": "user", "content": user_prompt},\n            ],\n            "temperature": agent.temperature,\n        }\n\n        if (\n            self.use_response_format\n            and agent.response_format\n            and agent.response_format.lower() == "json_object"\n        ):\n            kwargs["response_format"] = {"type": "json_object"}\n\n        last_error = None\n\n        for attempt in range(1, self.max_retries + 1):\n            try:\n                print(\n                    f"       ↳ LLM 请求(chat): provider={self.provider}, "\n                    f"agent={agent.agent_id}, model={self.model}, "\n                    f"timeout={self.timeout_seconds}s, attempt={attempt}/{self.max_retries}"\n                )\n                response = self.client.chat.completions.create(**kwargs)\n                return response.choices[0].message.content or ""\n\n            except Exception as e:\n                last_error = e\n\n                if not self._is_retryable_error(e):\n                    raise\n\n                wait = min(5 * (2 ** (attempt - 1)), 90)\n                print(\n                    f"⚠️ LLM 网络/超时错误，{wait}s 后重试 "\n                    f"({attempt}/{self.max_retries}): {e.__class__.__name__}: {e}"\n                )\n                time.sleep(wait)\n\n        raise RuntimeError(f"LLM chat 调用多次重试后仍失败: {last_error}")\n\n    def _dashscope_generation_chat(self, agent: MarkdownAgent, user_prompt: str) -> str:\n        """\n        qwen-deep-research 专用调用方式。\n\n        重要：\n        - qwen-deep-research 目前只支持 stream=True。\n        - 不走 OpenAI-compatible chat.completions。\n        - 不走 OpenAI-compatible /responses。\n        - web_search / web_extractor 是模型内部 DeepResearch 阶段能力，不在这里手动传 tools。\n        """\n        # 把 system 和 user 合并进 user content，更稳。\n        # DashScope Generation.call 对 qwen-deep-research 的 system 支持可能与普通 chat 不完全一致。\n        merged_prompt = (\n            f"【系统角色】\\n{agent.system}\\n\\n"\n            f"【任务要求】\\n"\n            f"请直接完成研究与分析，不要向用户反问确认。\\n"\n            f"如果需要搜索网页，请使用模型内置 DeepResearch 能力。\\n"\n            f"如果要求 JSON，请最终只输出可解析的 JSON，不要包裹 markdown 代码块。\\n\\n"\n            f"{user_prompt}"\n        )\n\n        messages = [\n            {"role": "user", "content": merged_prompt}\n        ]\n\n        last_error = None\n\n        for attempt in range(1, self.max_retries + 1):\n            try:\n                print(\n                    f"       ↳ LLM 请求(dashscope_generation): provider={self.provider}, "\n                    f"agent={agent.agent_id}, model={self.model}, stream=True, "\n                    f"attempt={attempt}/{self.max_retries}"\n                )\n\n                responses = self.dashscope.Generation.call(\n                    api_key=self.api_key,\n                    model=self.model,\n                    messages=messages,\n                    stream=True,\n                )\n\n                return self._process_dashscope_deep_research_stream(\n                    responses=responses,\n                    agent_id=agent.agent_id,\n                )\n\n            except Exception as e:\n                last_error = e\n\n                if not self._is_retryable_error(e):\n                    raise\n\n                wait = min(5 * (2 ** (attempt - 1)), 90)\n                print(\n                    f"⚠️ DashScope DeepResearch 网络/超时错误，{wait}s 后重试 "\n                    f"({attempt}/{self.max_retries}): {e.__class__.__name__}: {e}"\n                )\n                time.sleep(wait)\n\n        raise RuntimeError(f"DashScope qwen-deep-research 多次重试后仍失败: {last_error}")\n\n    def _process_dashscope_deep_research_stream(self, responses: Any, agent_id: str) -> str:\n        current_phase = None\n        phase_content = ""\n        answer_content_parts: List[str] = []\n        all_content_parts: List[str] = []\n        references: List[Dict[str, Any]] = []\n        web_sites: List[Dict[str, Any]] = []\n        keepalive_shown = False\n\n        for response in responses:\n            status_code = getattr(response, "status_code", 200)\n            if status_code != 200:\n                code = getattr(response, "code", "")\n                message = getattr(response, "message", "")\n                request_id = ""\n                try:\n                    request_id = response.get("request_id", "")\n                except Exception:\n                    pass\n                raise RuntimeError(\n                    f"DashScope 返回错误: HTTP {status_code}, code={code}, "\n                    f"message={message}, request_id={request_id}"\n                )\n\n            output = getattr(response, "output", None)\n            if not output:\n                continue\n\n            message_obj = output.get("message", {}) if isinstance(output, dict) else {}\n            phase = message_obj.get("phase")\n            content = message_obj.get("content", "") or ""\n            status = message_obj.get("status")\n            extra = message_obj.get("extra", {}) or {}\n\n            if phase != current_phase:\n                if current_phase and phase_content:\n                    print(f"\\n       ✓ {agent_id}: {current_phase} 阶段完成")\n                current_phase = phase\n                phase_content = ""\n                keepalive_shown = False\n\n                if phase:\n                    print(f"\\n       ↳ {agent_id}: 进入 {phase} 阶段")\n\n            if phase == "answer":\n                deep_research_extra = extra.get("deep_research", {}) if isinstance(extra, dict) else {}\n                new_refs = deep_research_extra.get("references") or []\n                if new_refs and new_refs != references:\n                    references = new_refs\n                    print(f"\\n       ↳ {agent_id}: 引用来源 {len(references)} 个")\n\n            if phase == "WebResearch":\n                deep_research_extra = extra.get("deep_research", {}) if isinstance(extra, dict) else {}\n                research_info = deep_research_extra.get("research") or {}\n                if status in ("streamingThinking", "streamingQueries"):\n                    goal = research_info.get("researchGoal")\n                    if goal:\n                        print(f"\\n       ↳ {agent_id}: 研究目标片段: {goal[:200]}", end="", flush=True)\n                elif status == "streamingWebResult":\n                    sites = research_info.get("webSites") or []\n                    if sites and sites != web_sites:\n                        web_sites = sites\n                        print(f"\\n       ↳ {agent_id}: 找到网页来源 {len(web_sites)} 个")\n                elif status == "WebResultFinished":\n                    print(f"\\n       ↳ {agent_id}: 网络搜索阶段完成")\n\n            if content:\n                phase_content += content\n                all_content_parts.append(content)\n\n                # 最终答案通常在 answer 阶段。\n                if phase == "answer" or phase is None:\n                    answer_content_parts.append(content)\n\n                # 避免日志刷屏，只显示很短的流式片段。\n                if os.getenv("DEEP_RESEARCH_VERBOSE_STREAM", "0") == "1":\n                    print(content, end="", flush=True)\n\n            if status and status != "typing":\n                if status == "streamingThinking":\n                    print(f"\\n       ↳ {agent_id}: 正在拆解研究任务并总结网页内容")\n                elif status == "streamingQueries":\n                    print(f"\\n       ↳ {agent_id}: 正在生成搜索查询")\n                elif status == "streamingWebResult":\n                    print(f"\\n       ↳ {agent_id}: 正在搜索和阅读网页")\n                elif status == "WebResultFinished":\n                    print(f"\\n       ↳ {agent_id}: WebResearch 完成")\n                elif status == "finished":\n                    usage = getattr(response, "usage", None)\n                    if usage:\n                        print(f"\\n       ↳ {agent_id}: token 使用: {usage}")\n                elif phase == "KeepAlive":\n                    if not keepalive_shown:\n                        print(f"\\n       ↳ {agent_id}: KeepAlive，等待下一阶段")\n                        keepalive_shown = True\n\n        if current_phase and phase_content:\n            print(f"\\n       ✓ {agent_id}: {current_phase} 阶段完成")\n\n        final_text = "".join(answer_content_parts).strip()\n        if not final_text:\n            final_text = "".join(all_content_parts).strip()\n\n        # 附加引用信息到文本末尾，方便后续原始记录保存；JSON 解析时 safe_json_loads 会提取 JSON 主体。\n        if references and os.getenv("APPEND_DEEP_RESEARCH_REFERENCES", "1") == "1":\n            refs_md = "\\n\\n<!-- DeepResearch References\\n"\n            for i, ref in enumerate(references, 1):\n                title = ref.get("title", "")\n                url = ref.get("url", "")\n                desc = ref.get("description", "")\n                refs_md += f"{i}. {title} | {url} | {desc}\\n"\n            refs_md += "-->\\n"\n            final_text += refs_md\n\n        return final_text'


def find_class_region(text: str) -> tuple[int, int]:
    start_match = re.search(r"^class\s+DeepResearchLLM\s*:", text, flags=re.MULTILINE)
    if not start_match:
        raise RuntimeError("没有找到 class DeepResearchLLM。")

    start = start_match.start()
    end_match = re.search(r"^def\s+build_records_context\s*\(", text[start:], flags=re.MULTILINE)
    if end_match:
        return start, start + end_match.start()

    raise RuntimeError("找到了 DeepResearchLLM，但没有找到后面的 build_records_context。")


def patch_python_file() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"没有找到 {TARGET}，请把补丁放到项目根目录运行。")

    text = TARGET.read_text(encoding="utf-8", errors="ignore")

    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")

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
        "deep_research_provider": "dashscope",
        "deep_research_model": "qwen-deep-research",
        "deep_research_api_mode": "dashscope_generation",
        "deep_research_tools": ["web_search", "web_extractor"],
        "llm_api_mode": "dashscope_generation",
        "use_response_format": False,
        "include_temperature": False,
        "background": False
    })

    PROVIDER_CONFIG.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    patch_python_file()
    patch_provider_config()

    print("✅ 已切换为 DashScope SDK qwen-deep-research 流式调用")
    print(f"   已备份: {BACKUP}")
    print(f"   已修改: {TARGET}")
    print(f"   已更新: {PROVIDER_CONFIG}")
    print()
    print("请确认已安装 dashscope：")
    print("   pip install -U dashscope")
    print()
    print("建议先小批量测试：")
    print("   set DASHSCOPE_API_KEY=你的key")
    print("   set LLM_TIMEOUT_SECONDS=600")
    print("   python deep_research_literature_agent.py --max-results 2 --batch-size 1")
    print()
    print("如果想看完整流式内容：")
    print("   set DEEP_RESEARCH_VERBOSE_STREAM=1")


if __name__ == "__main__":
    main()
