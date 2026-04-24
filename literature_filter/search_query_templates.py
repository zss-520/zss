from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class QueryTemplates:
    amp_synonyms: str = '("antimicrobial peptide" OR "host defense peptide" OR "antibacterial peptide" OR "AMP")'
    ai_synonyms: str = '("deep learning" OR "machine learning" OR "neural network" OR transformer OR CNN OR BERT OR "graph neural network")'
    strict_model_task_synonyms: str = '(prediction OR identification OR classification OR discrimination OR screening)'
    strict_binary_synonyms: str = '(binary OR classifier OR classification)'
    benchmark_task_synonyms: str = '(benchmark OR dataset OR "gold standard" OR "comprehensive assessment")'
    exclude_terms: str = 'NOT (review OR survey OR generation OR generative OR design OR VAE OR GAN OR diffusion OR hemolytic OR anticancer OR antiviral)'

    def build(self) -> Dict[str, str]:
        pmc_model_query = (
            f'({self.amp_synonyms} AND {self.ai_synonyms} AND '
            f'{self.strict_model_task_synonyms} AND {self.strict_binary_synonyms}) '
            f'AND OPEN_ACCESS:y {self.exclude_terms}'
        )

        s2_model_query = (
            f'{self.amp_synonyms} AND {self.ai_synonyms} AND '
            f'{self.strict_model_task_synonyms} AND {self.strict_binary_synonyms} {self.exclude_terms}'
        )

        openalex_model_query = (
            f'{self.amp_synonyms} AND {self.ai_synonyms} AND '
            f'{self.strict_model_task_synonyms} AND code'
        )

        serpapi_model_query = (
            '"antimicrobial peptide" (classification OR identification OR prediction) '
            '(github OR code OR repository) -(review) -(survey) -(generation) -(design) -(VAE) -(GAN)'
        )

        benchmark_query = (
            f'{self.amp_synonyms} AND {self.ai_synonyms} AND {self.benchmark_task_synonyms}'
        )

        return {
            "pmc_model_query": pmc_model_query,
            "s2_model_query": s2_model_query,
            "openalex_model_query": openalex_model_query,
            "serpapi_model_query": serpapi_model_query,
            "benchmark_query": benchmark_query,
        }


if __name__ == "__main__":
    from pprint import pprint
    pprint(QueryTemplates().build())