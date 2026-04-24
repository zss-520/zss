from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List


@dataclass
class PaperTypeResult:
    paper_title: str = ""
    is_model_paper: bool = False
    is_benchmark_paper: bool = False
    is_dataset_paper: bool = False
    is_review_paper: bool = False
    is_generation_or_design_paper: bool = False
    is_binary_amp_model: bool = False
    has_open_code_signal: bool = False
    has_weights_signal: bool = False
    should_download_full_text: bool = False
    should_send_to_paper_analyst: bool = False
    reject_reason: str = ""
    confidence: float = 0.0
    evidence_spans: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


REVIEW_PATTERNS = [
    r"\breview\b",
    r"\bsurvey\b",
    r"\bcomprehensive assessment\b",
    r"\bbenchmarking study\b",
]

BENCHMARK_PATTERNS = [
    r"\bbenchmark\b",
    r"\bgold standard\b",
    r"\bdataset\b",
    r"\bcorpus\b",
    r"\bresource\b",
]

MODEL_PATTERNS = [
    r"\bclassifier\b",
    r"\bclassification\b",
    r"\bidentification\b",
    r"\bprediction\b",
    r"\bdiscrimination\b",
    r"\bscreening\b",
    r"\bdeep learning\b",
    r"\bmachine learning\b",
    r"\bcnn\b",
    r"\bbert\b",
    r"\btransformer\b",
    r"\bgraph attention\b",
    r"\bgat\b",
]

GENERATION_PATTERNS = [
    r"\bgeneration\b",
    r"\bgenerative\b",
    r"\bdesign\b",
    r"\bvae\b",
    r"\bgan\b",
    r"\bdiffusion\b",
]

BINARY_AMP_PATTERNS = [
    r"\bantimicrobial peptide\b",
    r"\bamp\b",
    r"\bhost defense peptide\b",
    r"\bantibacterial peptide\b",
]

CODE_PATTERNS = [
    r"github\.com",
    r"gitlab\.com",
    r"gitee\.com",
    r"zenodo\.org",
    r"code available",
    r"source code",
    r"implementation",
    r"web server",
]

WEIGHTS_PATTERNS = [
    r"pretrained",
    r"weights",
    r"checkpoint",
    r"model file",
    r"download weights",
]


def _find_matches(text: str, patterns: List[str]) -> List[str]:
    lowered = text.lower()
    hits: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 60)
            span = text[start:end].replace("\n", " ")
            hits.append(span.strip())
            if len(hits) >= 5:
                return hits
    return hits


def classify_paper_from_title_abstract(title: str, abstract: str) -> PaperTypeResult:
    text = f"{title}\n\n{abstract}".strip()
    lowered = text.lower()
    result = PaperTypeResult(paper_title=title)

    review_hits = _find_matches(text, REVIEW_PATTERNS)
    benchmark_hits = _find_matches(text, BENCHMARK_PATTERNS)
    model_hits = _find_matches(text, MODEL_PATTERNS)
    generation_hits = _find_matches(text, GENERATION_PATTERNS)
    amp_hits = _find_matches(text, BINARY_AMP_PATTERNS)
    code_hits = _find_matches(text, CODE_PATTERNS)
    weight_hits = _find_matches(text, WEIGHTS_PATTERNS)

    result.is_review_paper = bool(review_hits)
    result.is_benchmark_paper = bool(benchmark_hits)
    result.is_dataset_paper = bool(benchmark_hits and not model_hits)
    result.is_model_paper = bool(model_hits)
    result.is_generation_or_design_paper = bool(generation_hits)
    result.is_binary_amp_model = bool(amp_hits and model_hits)
    result.has_open_code_signal = bool(code_hits)
    result.has_weights_signal = bool(weight_hits)

    result.evidence_spans.extend(review_hits[:1])
    result.evidence_spans.extend(benchmark_hits[:1])
    result.evidence_spans.extend(model_hits[:2])
    result.evidence_spans.extend(generation_hits[:1])
    result.evidence_spans.extend(code_hits[:1])

    # Lightweight decision policy for stage-1 filtering.
    score = 0.0
    if result.is_model_paper:
        score += 0.35
    if result.is_binary_amp_model:
        score += 0.25
    if result.has_open_code_signal:
        score += 0.15
    if result.has_weights_signal:
        score += 0.05
    if result.is_review_paper:
        score -= 0.20
    if result.is_benchmark_paper and not result.is_model_paper:
        score -= 0.20
    if result.is_generation_or_design_paper:
        score -= 0.30

    result.confidence = max(0.0, min(1.0, round(score + 0.30, 2)))

    if result.is_review_paper:
        result.reject_reason = "review_or_survey"
    elif result.is_generation_or_design_paper:
        result.reject_reason = "generation_or_design_not_binary_classifier"
    elif result.is_benchmark_paper and not result.is_model_paper:
        result.reject_reason = "benchmark_or_dataset_only"
    elif not result.is_model_paper:
        result.reject_reason = "no_clear_model_signal"
    elif not result.is_binary_amp_model:
        result.reject_reason = "not_clear_binary_amp_model"
    else:
        result.reject_reason = ""

    result.should_download_full_text = (
        result.is_model_paper
        and result.is_binary_amp_model
        and not result.is_generation_or_design_paper
        and not result.is_review_paper
        and result.confidence >= 0.55
    )
    result.should_send_to_paper_analyst = result.should_download_full_text

    return result


def build_classifier_prompt(title: str, abstract: str) -> str:
    return f"""
你是文献一级筛选器。请只根据标题和摘要，把论文分类为：
- model_paper
- benchmark_paper
- dataset_paper
- review_paper
- generation_or_design_paper

任务目标：只保留“纯 AMP 二分类判别模型论文”。
拒绝项：综述、benchmark-only、dataset-only、生成式/设计类、非 AMP、非二分类。
同时判断是否存在开源代码信号和预训练权重信号。

请严格输出 JSON 对象，字段如下：
{{
  "paper_title": "",
  "is_model_paper": true,
  "is_benchmark_paper": false,
  "is_dataset_paper": false,
  "is_review_paper": false,
  "is_generation_or_design_paper": false,
  "is_binary_amp_model": true,
  "has_open_code_signal": true,
  "has_weights_signal": false,
  "should_download_full_text": true,
  "should_send_to_paper_analyst": true,
  "reject_reason": "",
  "confidence": 0.91,
  "evidence_spans": ["...", "..."]
}}

标题：{title}
摘要：{abstract}
""".strip()


def classify_batch(records: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results = []
    for record in records:
        title = record.get("title", "")
        abstract = record.get("abstract", "")
        results.append(classify_paper_from_title_abstract(title, abstract).to_dict())
    return results


if __name__ == "__main__":
    demo = {
        "title": "AMP-BERT: A BERT-based model for antimicrobial peptide prediction",
        "abstract": "We present a binary classifier for antimicrobial peptide identification. Code is available at https://github.com/example/repo and pretrained weights are provided.",
    }
    print(json.dumps(classify_batch([demo]), ensure_ascii=False, indent=2))