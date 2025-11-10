# Orchestrate: Harvest → Enrich → Read → Analyze → Summarize → QA
# Keep comments in English.
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import google_search

generate_scholar_query = Agent(
    name="scholar_query_builder",
    model="gemini-2.5-flash-lite",
    description="Generate a precise Scholar/Google query for latest, high-impact AI papers.",
    instruction="""
You must output STRICT JSON with the key "scholar_query".
Task:
1) Identify trending AI subfields (e.g., "large language models", multimodal, RL, interpretability).
2) Bias to recent & high-quality sources: (2025 OR 2024), "survey", "benchmark", NeurIPS, ICLR, ICML, site:arxiv.org, site:openreview.net, filetype:pdf.
3) Construct ONE concise query string using quotes and Boolean operators.

    """.strip(),
    tools=[google_search],
    output_key="scholar_query"
)

results_harvester = Agent(
    name="results_harvester",
    model="gemini-2.5-flash-lite",
    description="Harvest candidate papers from a Scholar-style query.",
    instruction="""
You will receive a variable named {scholar_query} in context.
Use google_search to retrieve ~20 results for that query.
Deduplicate by title/URL; prefer arXiv/OpenReview/publisher.

Output:
{
  "candidates": [
    {"title":"...", "url":"...", "venue_hint": null, "year_hint": 2025}
  ]
}
    """.strip(),
    tools=[google_search],
    output_key="candidates"
)

metadata_enricher = Agent(
    name="metadata_enricher",
    model="gemini-2.5-flash-lite",
    description="Resolve canonical metadata (title, authors, year, venue, ids).",
    instruction="""
You will receive {candidates} in context.

For each item, resolve: title, authors, year, venue, doi/arxiv_id, best pdf_url, abstract (if available).
Output:
{
  "papers_meta": [
    {
      "title":"...", "authors":["..."], "year":2025, "venue":"...", "doi": null,
      "arxiv_id": null, "pdf_url":"...", "abstract":"..."
    }
  ]
}
    """.strip(),
    tools=[],
    output_key="papers_meta"
)

pdf_text_reader = Agent(
    name="pdf_text_reader",
    model="gemini-2.5-flash-lite",
    description="Extract sectioned text from PDF URLs.",
    instruction="""
You will receive {papers_meta} in context.
For each paper with a reachable pdf_url, extract sectioned text and figure/table captions.

Output ONLY:
{
  "papers_full": [
    {
      "title":"...",
      "sections":[{"heading":"Introduction","text":"..."},{"heading":"Method","text":"..."}],
      "figures_tables":[{"caption":"..."}]
    }
  ]
}
If a PDF is unreachable, skip it (do not fail the batch).
    """.strip(),
    tools=[],
    output_key="papers_full"
)

paper_analyzer = Agent(
    name="paper_analyzer",
    model="gemini-2.5-flash-lite",
    description="Analyze claims, methods, experiments, limitations; score novelty & reliability.",
    instruction="""
You will receive {papers_full} in context.

Output ONLY:
{
  "analysis": [
    {
      "title":"...",
      "type":"survey|method|benchmark|dataset|application|position",
      "claims":[{"claim":"...","evidence_snippet":"<=30 words"}],
      "method_summary":"<=80 words",
      "experiments":[{"dataset":"...","metric":"...","result":"..."}],
      "limitations":["..."],
      "novelty_score_0_5": 0,
      "reliability_risk_0_5": 0
    }
  ]
}
    """.strip(),
    tools=[],
    output_key="analysis"
)

final_summarizer = Agent(
    name="final_summarizer",
    model="gemini-2.5-flash-lite",
    description="Produce crisp TL;DR summaries for decision-makers and builders.",
    instruction="""
You will receive {papers_meta} and {analysis} in context.

Output ONLY:
{
  "summaries":[
    {
      "title":"...",
      "tldr_5_bullets":["...","...","...","...","..."],
      "who_should_read":["role1","role2","role3"],
      "implementation_notes":["note1","note2","note3"],
      "key_citations":["ref1","ref2","ref3"],
      "pdf_url":"..."
    }
  ]
}
    """.strip(),
    tools=[],
    output_key="summaries"
)

qa_gate = Agent(
    name="qa_gate",
    model="gemini-2.5-flash-lite",
    description="Validate required fields, alignment claims↔evidence, word limits, dedupe.",
    instruction="""
You will receive {summaries} in context.

Validate and output ONLY:
{"ok": true, "issues": []}
or
{"ok": false, "issues": ["...","..."]}
    """.strip(),
    tools=[],
    output_key="qa_report"
)

root_agent = SequentialAgent(
    name="ai_papers_pipeline",
    description="Sequential pipeline for discovering and summarizing latest AI papers.",
    sub_agents=[
        generate_scholar_query,
        results_harvester,
        metadata_enricher,
        pdf_text_reader,
        paper_analyzer,
        final_summarizer,
        qa_gate
    ],
)
