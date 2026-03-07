from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.evidence import build_evidence_collection_node
from app.graph.nodes.intake import build_intake_node
from app.graph.nodes.retrieval import build_retrieval_node
from app.graph.nodes.safety_guard import build_safety_guard_node
from app.graph.nodes.ticket import build_ticket_creation_node
from app.graph.nodes.troubleshooting import build_troubleshooting_node
from app.graph.state import SupportGraphState


@dataclass(slots=True)
class WorkflowDependencies:
    llm_client: object
    retrieval_service: object
    validation_service: object
    ticket_service: object
    retrieval_top_k: int


def build_workflow(dependencies: WorkflowDependencies):
    graph = StateGraph(SupportGraphState)

    graph.add_node("intake", build_intake_node(dependencies.llm_client))
    graph.add_node("safety_guardrails", build_safety_guard_node())
    graph.add_node("retrieval", build_retrieval_node(dependencies.retrieval_service, dependencies.retrieval_top_k))
    graph.add_node(
        "troubleshooting",
        build_troubleshooting_node(dependencies.llm_client, dependencies.validation_service),
    )
    graph.add_node("evidence_collection", build_evidence_collection_node())
    graph.add_node("ticket_creation", build_ticket_creation_node(dependencies.ticket_service))

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        _route_after_intake,
        {
            "safety_guardrails": "safety_guardrails",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "safety_guardrails",
        _route_after_safety,
        {
            "retrieval": "retrieval",
            "evidence_collection": "evidence_collection",
        },
    )
    graph.add_edge("retrieval", "troubleshooting")
    graph.add_conditional_edges(
        "troubleshooting",
        _route_after_troubleshooting,
        {
            "end": END,
            "evidence_collection": "evidence_collection",
        },
    )
    graph.add_conditional_edges(
        "evidence_collection",
        _route_after_evidence,
        {
            "end": END,
            "ticket_creation": "ticket_creation",
        },
    )
    graph.add_edge("ticket_creation", END)
    return graph.compile()


def _route_after_safety(state: SupportGraphState) -> str:
    if state.get("safety_assessment", {}).get("escalate_immediately"):
        return "evidence_collection"
    intent = state.get("classification", {}).get("intent")
    request_ticket = state.get("request", {}).get("request_ticket", False)
    if intent == "escalate" or request_ticket:
        return "evidence_collection"
    return "retrieval"


def _route_after_intake(state: SupportGraphState) -> str:
    if state.get("system_message"):
        return "end"
    return "safety_guardrails"


def _route_after_troubleshooting(state: SupportGraphState) -> str:
    next_action = state.get("next_action")
    if next_action in {"collect_evidence", "escalate"}:
        return "evidence_collection"
    return "end"


def _route_after_evidence(state: SupportGraphState) -> str:
    if state.get("missing_fields"):
        return "end"
    return "ticket_creation"
