

from langgraph.graph import StateGraph, START, END

from state import SupportState
from memory_db import get_checkpointer
from nodes import (
    classify_intent_node,
    route_by_intent,
    route_after_department,
    sales_agent_node,
    technical_agent_node,
    billing_agent_node,
    account_agent_node,
    human_approval_node,
    supervisor_node,
    make_memory_recall_node,
)


def build_graph():
    checkpointer = get_checkpointer()
    memory_recall_node = make_memory_recall_node(checkpointer)

    workflow = StateGraph(SupportState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("sales_agent", sales_agent_node)
    workflow.add_node("technical_agent", technical_agent_node)
    workflow.add_node("billing_agent", billing_agent_node)
    workflow.add_node("account_agent", account_agent_node)
    workflow.add_node("memory_recall", memory_recall_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("supervisor", supervisor_node)

    workflow.add_edge(START, "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent",
            "memory_recall": "memory_recall",
        },
    )

    # each department agent then checks whether human approval is needed
    for dept_node in ["sales_agent", "technical_agent", "billing_agent", "account_agent"]:
        workflow.add_conditional_edges(
            dept_node,
            route_after_department,
            {
                "human_approval": "human_approval",
                "supervisor": "supervisor",
            },
        )

    workflow.add_edge("human_approval", "supervisor")
    workflow.add_edge("supervisor", END)
    workflow.add_edge("memory_recall", END)

    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"],
    )

    return graph, checkpointer
