# Minimal pipeline with a refinement loop (readable text output, no greetings).
# Query → Harvest/Enrich → Summarize → (Critic ↔ Refiner) x3 → Output

from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.tools import google_search, FunctionTool

# 1) Build a strong Scholar/Google query (text key, no greetings)
scholar_query_builder = Agent(
    name="scholar_query_builder",
    model="gemini-2.5-flash-lite",
    description="Build an optimized Scholar/Google query for latest high-impact AI papers.",
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

# 2) Harvest + light enrich (keep as JSON for internal passing)
harvest_enrich = Agent(
    name="harvest_enrich",
    model="gemini-2.5-flash-lite",
    description="Search and return enriched top results.",
    instruction="""
The scholar query:  {scholar_query}. 
1) Use google_search to fetch ~20 results for scholar_query.
2) Deduplicate by title/url; prefer arXiv/OpenReview/publisher; keep the best 8–12.
3) For each: title, url, pdf_url if visible, venue_hint, year_hint, abstract_hint (short).

Output ONLY:
{
  "papers_meta": [
    {
      "title":"...", "url":"...", "pdf_url":"nullable",
      "venue_hint":"nullable", "year_hint":2025, "abstract_hint":"nullable"
    }
  ]
}
    """.strip(),
    tools=[google_search],
    output_key="papers_meta"
)

# 3) Initial readable summary (Markdown, no greetings)
initial_summarizer = Agent(
    name="initial_summarizer",
    model="gemini-2.5-flash-lite",
    description="Produce a concise, readable digest in Markdown.",
    instruction="""
Meta data about the papers: {papers_meta}. 

Write a compact Markdown digest (no greeting, no preface). Include:
- H1: Latest AI Papers Digest
- For the top 6–8 papers, per paper:
  - H2: <Title> (Year if known)
  - One sentence with venue/source and link (prefer pdf_url if present, else url).
  - 3–5 bullets: key idea, evidence/benchmarks (if visible), notable limitation (if visible), why it matters.

Constraints:
- Keep it tight and readable.
- Do not fabricate details; if unknown, omit.
- Output ONLY the digest text (no JSON).
    """.strip(),
    tools=[],
    output_key="current_summary"
)

# --- Refinement loop: critic ↔ refiner (max 3 iterations) ---

critic_agent = Agent(
    name="critic_agent",
    model="gemini-2.5-flash-lite",
    description="Critique the digest constructively.",
    instruction="""
Review the digest below and improve its usefulness and clarity.
Digest:
{current_summary}

Rules:
- If the digest is already excellent and complete, respond EXACTLY with: APPROVED
- Otherwise, provide 2–4 SPECIFIC, actionable suggestions AND up to 3 clarifying questions
  to make the digest tighter, more accurate, or more useful (e.g., clearer links, tighter TL;DRs,
  remove fluff, ensure consistent structure).

Output ONLY the critique text or the single token APPROVED.
    """.strip(),
    output_key="critique"
)

def exit_loop():
    """Call this ONLY when the critique is 'APPROVED'."""
    return {"status": "approved"}

refiner_agent = Agent(
    name="refiner_agent",
    model="gemini-2.5-flash-lite",
    description="Refine the digest based on the critique; stop when approved.",
    instruction="""
You have:
Digest:
{current_summary}

Critique:
{critique}

Behavior:
- IF critique is EXACTLY "APPROVED", call the exit_loop tool and do nothing else.
- OTHERWISE, revise the digest to fully address the suggestions and questions:
  * tighten wording, remove fluff
  * ensure consistent structure per paper (Title, link line, 3–5 bullets)
  * add/clarify links only if present in the source list (do not invent)
  * keep Markdown, no greetings, no meta commentary

Output ONLY the revised digest text (no JSON).
    """.strip(),
    tools=[FunctionTool(exit_loop)],
    output_key="current_summary"
)

refinement_loop = LoopAgent(
    name="summary_refinement_loop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=3  # hard limit
)

# 4) Root: run all steps in order; final output is readable text in "current_summary"
root_agent = SequentialAgent(
    name="ai_papers_min_pipeline_with_refinement",
    description="Query → Harvest/Enrich → Summarize → Critic/Refiner loop (x3).",
    sub_agents=[
        scholar_query_builder,
        harvest_enrich,
        initial_summarizer,
        refinement_loop
    ],
)
