"""
agents/query_architect_prompt.py — Improved query generation prompts.

Generates primary-data-focused queries in tiered categories:
  1. Compound/entity specific (exact names, IDs)
  2. Assay/measurement specific (IC50, DAR, yield, kinetics)
  3. Methodology (synthesis, conjugation, characterisation)
  4. Database/repository (PubMed, ChEMBL, ClinicalTrials, BindingDB)
  5. Column-targeted (one query per high-priority column)

Also supports domain-context injection from the user's chatbot input.
"""

from __future__ import annotations

import json

DEFAULT_NUM_QUERIES = 15


QUERY_ARCHITECT_SYSTEM = """\
You are a precision scientific literature search specialist.
Your ONLY job is to generate search queries that retrieve PRIMARY EXPERIMENTAL papers
— papers where the authors ran experiments themselves and report raw measured data.

STRICT RULES:
1. NEVER generate queries for review articles, meta-analyses, surveys, or editorials.
2. Every query must target papers that MEASURED or SYNTHESISED or TESTED something.
3. Prefer queries that return structured tabular data (IC50, binding affinities, yields, DAR, etc.).
4. Prioritise: PubMed, ACS, RSC, Elsevier, Nature, Wiley, ClinicalTrials.gov, ChEMBL.
5. Include at least 2 database-specific queries using site: operator.
6. Use field-value syntax: specific assays, instrument names, measured quantities.
7. Each query must be meaningfully different — cover different facets.
8. NEVER add quotes around entire queries — only around exact phrases.
9. Return ONLY a JSON object with key "search_queries": [...]. No prose, no markdown.
"""


def build_query_prompt(
    topic: str,
    columns: list[str],
    num_queries: int = DEFAULT_NUM_QUERIES,
    existing_queries: list[str] | None = None,
    domain_context: str = "",
) -> str:
    col_str    = ", ".join(f'"{c}"' for c in columns[:20]) if columns else "general research data"
    avoid_str  = ""
    if existing_queries:
        avoid_str = (
            "\nAVOID repeating these already-used queries:\n"
            + "\n".join(f"  - {q}" for q in existing_queries[:10])
        )

    domain_block = ""
    if domain_context:
        domain_block = (
            f"\n\nUSER'S DOMAIN CONTEXT (use this to sharpen queries):\n{domain_context}"
        )

    return f"""Generate {num_queries} highly targeted search queries for finding PRIMARY EXPERIMENTAL papers.

TOPIC: "{topic}"{domain_block}

TARGET COLUMNS (each query should help fill at least one of these):
{col_str}

REQUIRED QUERY DISTRIBUTION:
• 3-4 compound/entity-specific queries (target exact known entities, approved drugs, gene names)
• 3-4 assay/measurement queries (IC50, EC50, DAR, yield, kd, binding affinity, cell viability)
• 2-3 methodology queries (synthesis, conjugation, characterisation, in vitro, in vivo)
• 2-3 database queries using site: (site:pubmed.ncbi.nlm.nih.gov, site:pubs.acs.org, etc.)
• 2-3 column-targeted queries (one per most important column — include units or format hints)

QUALITY RULES:
- Every query must target papers with MEASURED DATA, not reviews
- Use Boolean operators AND/OR where they help precision
- For biopharma/chemistry topics: include relevant assay names, units (nM, µM, %EE)
- For biology topics: include organism, cell line, assay type
- Prefer filetype:pdf when searching for papers
- Include at least one query using 2024 or 2025 for recent data{avoid_str}

Return ONLY:
{{"search_queries": ["query 1", "query 2", ...]}}"""


def build_fallback_prompt(
    topic: str,
    gap_columns: list[str],
    existing_queries: list[str],
    num_queries: int = 5,
    domain_context: str = "",
) -> str:
    gap_str   = ", ".join(f'"{c}"' for c in gap_columns)
    exist_str = "\n".join(f"  - {q}" for q in existing_queries[:10])
    dom_block = f"\nDomain context: {domain_context}" if domain_context else ""

    return f"""The dataset for "{topic}" has HIGH NULL RATES in these columns: {gap_str}.{dom_block}

Previous queries that didn't fill these gaps:
{exist_str}

Generate {num_queries} NEW search queries SPECIFICALLY targeting the missing columns.
Each query must be completely different from the existing ones.
Focus on finding papers that explicitly report values for: {gap_str}

Examples of targeted queries:
- For "DAR" gaps: "antibody drug conjugate drug-to-antibody ratio measured HIC-HPLC"
- For "SMILES" gaps: "compound synthesis characterization SMILES NMR site:pubs.acs.org"
- For "IC50" gaps: "cytotoxicity assay IC50 nM table results filetype:pdf"

Return ONLY:
{{"fallback_queries": ["query 1", "query 2", ...]}}"""
