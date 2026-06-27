"""Gradio UI for the Scientific Literature Search system.

A search box → ranked results (Markdown) + a facet/cluster sidebar + the agent's
decision log. ``gradio`` is imported lazily.
"""

from __future__ import annotations

from typing import Optional

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_ui(cfg: Optional[AppConfig] = None):
    import gradio as gr  # lazy
    from ..agent.search_agent import SearchAgent

    cfg = cfg or AppConfig()
    agent = SearchAgent(cfg, load_model=True)

    def do_search(query):
        if not query.strip():
            return "Enter a query.", "", ""
        job = agent.search(query, save=False)
        sd = job.to_dict()
        md = [f"**{len(sd['results'])} results** · intent: `{sd['intent']}`"]
        for r in sd["results"]:
            cats = " ".join(r["categories"][:3])
            md.append(f"\n**{r['rank']}. {r['title']}**  \n`{cats}` · score {r['score']:.3f}  \n{r['abstract']}")
        facets = "### Fields\n" + "\n".join(f"- {f['name']} ({f['count']})" for f in sd["facets"])
        clusters = "### Topic clusters\n" + "\n".join(f"- {c['label']} ({len(c['paper_ids'])})" for c in sd["clusters"])
        sugg = "### Explore\n" + "\n".join(f"- {s}" for s in sd["suggestions"])
        dec = "Agent: " + " · ".join(f"{d['id']}={d['branch']}" for d in sd["decisions"])
        return "\n".join(md), f"{facets}\n\n{clusters}\n\n{sugg}", dec

    with gr.Blocks(title=cfg.project_title) as demo:
        gr.Markdown(f"# 🔎 {cfg.project_title}\nHybrid (dense + BM25) paper search with facets, topic "
                    "clusters and related papers — driven by an agentic query-understanding pipeline.")
        with gr.Row():
            q = gr.Textbox(label="Query", scale=4,
                           value="contrastive learning for sentence embeddings")
            btn = gr.Button("Search", variant="primary", scale=1)
        dec = gr.Markdown()
        with gr.Row():
            results = gr.Markdown(label="Results")
            sidebar = gr.Markdown(label="Explore")
        btn.click(do_search, [q], [results, sidebar, dec])
        q.submit(do_search, [q], [results, sidebar, dec])
    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False) -> None:
    build_ui().launch(server_name=server_name, server_port=server_port, share=share)


__all__ = ["build_ui", "launch"]
