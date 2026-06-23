"""Read-only Q&A capability — wraps the existing NL→SPARQL ask pipeline.

This is the only capability that needs no plan/confirm round-trip: a question
does not mutate the graph, so the agent answers immediately. The planner
special-cases the ``question`` intent and calls :meth:`QueryCapability.answer`
directly. We still register it as a capability (so ``get_capabilities()`` is the
single source of truth for what the agent can do, and the classifier prompt can
include its ``describe()`` line), and we still implement ``plan``/``execute`` so
it satisfies the protocol: ``plan`` emits a single no-write ``answer`` step and
``execute`` fulfils it by delegating to :meth:`answer`.

Reuses :class:`cograph_client.nlp.pipeline.NLQueryPipeline.ask` — the exact same
engine the ``/ask`` route calls — so the agent and the legacy route share one
Q&A implementation (no divergence).
"""

from __future__ import annotations

from cograph_client.agent.registry import AgentContext, PlanStep
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri


class QueryCapability:
    name = "query"

    def describe(self) -> str:
        return (
            "Answer a read-only question about the data in the knowledge graph "
            "(counts, lookups, relationships) by generating and running SPARQL. "
            "Use for any 'how many', 'which', 'what', 'list', 'show me' question."
        )

    async def answer(self, ctx: AgentContext, question: str) -> dict:
        """Run the ask pipeline and return ``{answer, sparql, rows, narrative}``.

        Builds the pipeline the same way ``api/routes/ask.py`` does: ontology
        from the tenant graph, instance data from the KG-specific graph.
        """
        pipeline = self._build_pipeline(ctx)
        ontology_graph = tenant_graph_uri(ctx.tenant_id)
        instance_graph = (
            kg_graph_uri(ctx.tenant_id, ctx.kg_name) if ctx.kg_name else ontology_graph
        )
        result = await pipeline.ask(question, ontology_graph, instance_graph)
        return {
            "answer": result.answer,
            "sparql": result.sparql,
            "narrative": getattr(result, "narrative_answer", ""),
            # The pipeline does not surface raw rows on NLResult; the formatted
            # answer + sparql are what callers render. Keep the key present (empty)
            # so the contract is stable for clients that look for it.
            "rows": [],
        }

    def _build_pipeline(self, ctx: AgentContext):
        # Lazy import so importing the agent registry never drags in the heavy
        # pipeline module (and its anthropic client) at app-boot registration.
        from cograph_client.nlp.pipeline import NLQueryPipeline

        return NLQueryPipeline(ctx.neptune, ctx.anthropic_key)

    async def plan(self, ctx: AgentContext, instruction: str) -> list[PlanStep]:
        # A question is read-only: a single no-write step the planner can also
        # fast-path. confidence 1.0 — answering is always applicable to a
        # question; the planner decides whether the intent IS a question.
        return [
            PlanStep(
                capability=self.name,
                action="answer",
                params={"question": instruction},
                rationale="Read-only question; answer directly with SPARQL.",
                confidence=1.0,
                preview={"summary": "Runs a read-only SPARQL query; no writes."},
                cost={},
            )
        ]

    async def execute(self, ctx: AgentContext, step: PlanStep) -> dict:
        question = step.params.get("question", "")
        out = await self.answer(ctx, question)
        return {"kind": "answer", **out}
