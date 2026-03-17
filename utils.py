"""Contains useful utility functions."""

import json
import urllib.parse
from pathlib import Path

import requests
import tiktoken
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from constants import (
    DEFAULT_FINETUNING_EPOCHS,
    MODEL_TO_INPUT_PRICE_PER_TOKEN,
    MODEL_TO_OUTPUT_PRICE_PER_TOKEN,
    FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN,
    PUBMED_TOOL_NAME,
)
from prompts_vlab import format_references


def get_pubmed_central_article(pmcid: str, abstract_only: bool = False) -> tuple[str | None, list[str] | None]:
    text_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_JSON/PMC{pmcid}/unicode"
    response = requests.get(text_url)
    response.raise_for_status()

    try:
        article = response.json()
    except json.JSONDecodeError:
        return None, None

    document = article[0]["documents"][0]
    title = next(passage["text"] for passage in document["passages"] if passage["infons"]["section_type"] == "TITLE")

    passages = [passage for passage in document["passages"] if passage["infons"]["type"] in {"abstract", "paragraph"}]

    if abstract_only:
        passages = [passage for passage in passages if passage["infons"]["section_type"] in ["ABSTRACT"]]
    else:
        passages = [
            passage
            for passage in passages
            if passage["infons"]["section_type"] in ["ABSTRACT", "INTRO", "RESULTS", "DISCUSS", "CONCL", "METHODS"]
        ]

    content = [passage["text"] for passage in passages]
    return title, content


def run_pubmed_search(query: str, num_articles: int = 3, abstract_only: bool = False) -> str:
    print(
        f'Searching PubMed Central for {num_articles} articles ({"abstracts" if abstract_only else "full text"}) with query: "{query}"'
    )

    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={urllib.parse.quote_plus(query)}&retmax={2 * num_articles}&retmode=json&sort=relevance"
    response = requests.get(search_url)
    response.raise_for_status()
    pmcids_found = response.json()["esearchresult"]["idlist"]

    texts = []
    pmcids = []

    for pmcid in pmcids_found:
        if len(pmcids) >= num_articles:
            break

        title, content = get_pubmed_central_article(
            pmcid=pmcid,
            abstract_only=abstract_only,
        )

        if title is None:
            continue

        joined_content = "\n\n".join(content or [])
        texts.append(f"PMCID = {pmcid}\n\nTitle = {title}\n\n{joined_content}")
        pmcids.append(pmcid)

    article_count = len(texts)

    print(f"Found {article_count:,} articles on PubMed Central")

    if article_count == 0:
        combined_text = f'No articles found on PubMed Central for the query "{query}".'
    else:
        combined_text = format_references(
            references=tuple(texts),
            reference_type="paper",
            intro=f'Here are the top {article_count} articles on PubMed Central for the query "{query}":',
        )

    return combined_text


def run_tools(
    tool_calls: list[ChatCompletionMessageToolCall],
) -> tuple[list[str], list[ChatCompletionMessageParam]]:
    tool_outputs: list[str] = []
    tool_messages: list[ChatCompletionMessageParam] = []

    for tool_call in tool_calls:
        if tool_call.function.name == PUBMED_TOOL_NAME:
            args_dict = json.loads(tool_call.function.arguments)
            output = run_pubmed_search(**args_dict)
            tool_outputs.append(output)

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                }
            )
        else:
            raise ValueError(f"Unknown tool: {tool_call.function.name}")

    return tool_outputs, tool_messages


def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def update_token_counts(
    token_counts: dict[str, int],
    discussion: list[dict[str, str]],
    response: str,
) -> None:
    new_input_token_count = sum(count_tokens(turn["message"]) for turn in discussion)
    new_output_token_count = count_tokens(response)

    token_counts["input"] += new_input_token_count
    token_counts["output"] += new_output_token_count
    token_counts["max"] = max(token_counts["max"], new_input_token_count + new_output_token_count)


def count_discussion_tokens(
    discussion: list[dict[str, str]],
) -> dict[str, int]:
    token_counts = {
        "input": 0,
        "output": 0,
        "max": 0,
    }

    for index, turn in enumerate(discussion):
        if turn["agent"] != "User":
            update_token_counts(
                token_counts=token_counts,
                discussion=discussion[:index],
                response=turn["message"],
            )

    return token_counts


def _find_model_price_key(model: str, price_dict: dict[str, float]) -> str | None:
    if model in price_dict:
        return model

    matching_keys = [key for key in price_dict if model.startswith(key)]
    if matching_keys:
        return max(matching_keys, key=len)

    return None


def compute_token_cost(model: str, input_token_count: int, output_token_count: int) -> float:
    input_key = _find_model_price_key(model, MODEL_TO_INPUT_PRICE_PER_TOKEN)
    output_key = _find_model_price_key(model, MODEL_TO_OUTPUT_PRICE_PER_TOKEN)

    if input_key is None or output_key is None:
        raise ValueError(f'Cost of model "{model}" not known')

    return (
        input_token_count * MODEL_TO_INPUT_PRICE_PER_TOKEN[input_key]
        + output_token_count * MODEL_TO_OUTPUT_PRICE_PER_TOKEN[output_key]
    )


def print_cost_and_time(
    token_counts: dict[str, int],
    model: str,
    elapsed_time: float,
) -> None:
    print(f"Input token count: {token_counts['input']:,}")
    print(f"Output token count: {token_counts['output']:,}")
    print(f"Tool token count: {token_counts['tool']:,}")
    print(f"Max token length: {token_counts['max']:,}")

    try:
        cost = compute_token_cost(
            model=model,
            input_token_count=token_counts["input"] + token_counts["tool"],
            output_token_count=token_counts["output"],
        )
        print(f"Cost: ${cost:.2f}")
    except ValueError as e:
        print(f"Warning: {e}")

    print(f"Time: {int(elapsed_time // 60)}:{int(elapsed_time % 60):02d}")


def compute_finetuning_cost(model: str, token_count: int, num_epochs: int = DEFAULT_FINETUNING_EPOCHS) -> float:
    if model not in FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN:
        raise ValueError(f'Cost of model "{model}" not known')

    return token_count * FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN[model] * num_epochs


def get_summary(discussion: list[dict[str, str]]) -> str:
    return discussion[-1]["message"]


def load_summaries(discussion_paths: list[Path]) -> tuple[str, ...]:
    summaries = []
    for discussion_path in discussion_paths:
        with open(discussion_path, "r", encoding="utf-8") as file:
            discussion = json.load(file)
        summaries.append(get_summary(discussion))
    return tuple(summaries)


def save_meeting(save_dir: Path, save_name: str, discussion: list[dict[str, str]]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f"{save_name}.json", "w", encoding="utf-8") as f:
        json.dump(discussion, f, indent=4, ensure_ascii=False)

    with open(save_dir / f"{save_name}.md", "w", encoding="utf-8") as file:
        for turn in discussion:
            file.write(f"## {turn['agent']}\n\n{turn['message']}\n\n")