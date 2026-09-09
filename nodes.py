

import re
from langchain_core.messages import SystemMessage, HumanMessage

from llm import llm
from rag import rag_search
from memory_db import get_conversation_history

VALID_INTENTS = ["Sales", "Technical", "Billing", "Account", "Memory"]

# Any of these showing up in the raw query forces human approval,
# regardless of which department ends up handling it.
APPROVAL_KEYWORDS = [
    "refund", "cancel", "cancellation", "close my account", "account closure",
    "close the account", "compensation", "compensate", "escalate",
    "escalation", "speak to a manager", "speak to your manager", "supervisor",
]


def classify_intent_node(state):
    """Classify the customer's message into one of the five intents and
    flag whether it needs human approval before going out."""
    query = state["query"]

    system = SystemMessage(content=(
        "You are an intent classifier for a customer support system. "
        "Classify the customer's message into EXACTLY ONE of these categories: "
        "Sales, Technical, Billing, Account, Memory.\n\n"
        "- Sales: product information, subscription plans, pricing questions\n"
        "- Technical: application errors, installation issues, login problems, "
        "configuration or performance issues\n"
        "- Billing: invoices, payments, refunds\n"
        "- Account: password resets, profile updates, account activation, "
        "deactivation, account closure\n"
        "- Memory: the customer is asking about a PREVIOUS conversation or "
        "issue (e.g. 'what was my previous issue', 'what did I ask before')\n\n"
        "Reply with ONLY the single category word, nothing else."
    ))
    human = HumanMessage(content=query)

    response = llm.invoke([system, human])
    raw = response.content.strip()

    # find whichever valid intent word shows up in the reply
    intent = next((i for i in VALID_INTENTS if i.lower() in raw.lower()), None)

    # local models don't always return clean output, so fall back to a
    # keyword match if we couldn't parse a clean intent above
    if intent is None:
        q = query.lower()
        if any(w in q for w in ["previous", "last time", "before", "earlier"]):
            intent = "Memory"
        elif any(w in q for w in ["price", "plan", "pricing", "cost", "subscription"]):
            intent = "Sales"
        elif any(w in q for w in ["crash", "error", "install", "login", "bug", "not working"]):
            intent = "Technical"
        elif any(w in q for w in ["refund", "invoice", "payment", "bill", "charge"]):
            intent = "Billing"
        elif any(w in q for w in ["password", "account", "profile", "email"]):
            intent = "Account"
        else:
            intent = "Sales"  # safe default

    # approval check runs on the raw query text, independent of intent
    q_lower = query.lower()
    requires_approval = any(k in q_lower for k in APPROVAL_KEYWORDS)

    return {
        "intent": intent,
        "requires_approval": requires_approval,
        "approval_status": "pending" if requires_approval else "not_required",
    }


def route_by_intent(state):
    """Conditional edge selector: picks the next node based on classified intent."""
    intent = state["intent"]
    mapping = {
        "Sales": "sales_agent",
        "Technical": "technical_agent",
        "Billing": "billing_agent",
        "Account": "account_agent",
        "Memory": "memory_recall",
    }
    return mapping.get(intent, "sales_agent")


def route_after_department(state):
    """After a department agent drafts a response, decide if it needs sign-off."""
    if state.get("requires_approval"):
        return "human_approval"
    return "supervisor"


def _run_department_agent(state, department_name: str, role_description: str):
    """Shared logic behind all four department agents - only the persona
    and system prompt differ between them."""
    query = state["query"]
    context = rag_search(query, top_k=4)

    system = SystemMessage(content=(
        f"You are the {department_name} support agent for ABC Technologies. "
        f"{role_description}\n\n"
        "Use the following reference material from the company's knowledge base "
        "to answer accurately. If the reference material doesn't fully cover the "
        "question, answer using your best general knowledge of standard SaaS "
        "support practices, but prefer the reference material when relevant.\n\n"
        f"--- REFERENCE MATERIAL ---\n{context}\n--- END REFERENCE MATERIAL ---\n\n"
        "Write a clear, professional, concise response directly to the customer."
    ))
    human = HumanMessage(content=query)

    response = llm.invoke([system, human])

    return {
        "retrieved_context": context,
        "department_response": response.content.strip(),
    }


def sales_agent_node(state):
    return _run_department_agent(
        state, "Sales",
        "You help customers with product information, subscription plans, and pricing details."
    )


def technical_agent_node(state):
    return _run_department_agent(
        state, "Technical Support",
        "You help customers resolve application errors, installation issues, login "
        "problems, and configuration issues."
    )


def billing_agent_node(state):
    return _run_department_agent(
        state, "Billing",
        "You help customers with invoices, payment issues, and refund requests. "
        "Note: refund requests require human supervisor approval before being finalized - "
        "you should draft the response, but do not promise a final outcome."
    )


def account_agent_node(state):
    return _run_department_agent(
        state, "Account",
        "You help customers with password resets, profile updates, and account "
        "activation/deactivation, including account closure requests. Note: account "
        "closure requires human supervisor approval before being finalized - you "
        "should draft the response, but do not promise a final outcome."
    )


# The graph is compiled with interrupt_before=["human_approval"], so
# execution physically pauses right before this node runs whenever
# requires_approval is True. main.py handles that pause, asks a human
# for a decision, and writes it into approval_status via
# graph.update_state() before resuming - so by the time this function
# actually runs, a decision is already sitting in the state.
def human_approval_node(state):
    status = state.get("approval_status", "pending")

    if status == "approved":
        # response is kept as-is; LangGraph still needs at least one
        # key written back, so we just re-affirm the status
        return {"approval_status": "approved"}

    elif status == "rejected":
        notes = state.get("approval_notes", "")
        rejection_msg = (
            "After review, this request could not be approved at this time. "
            "A member of our support team will follow up with you directly "
            "regarding next steps."
        )
        if notes:
            rejection_msg += f" Note from supervisor: {notes}"
        return {"department_response": rejection_msg, "approval_status": "rejected"}

    else:
        # shouldn't normally land here - main.py only resumes after a
        # decision has been recorded
        return {"approval_status": "pending"}


def supervisor_node(state):
    """Final pass over the draft response - tightens tone/clarity without
    touching the underlying facts."""
    draft = state.get("department_response", "")
    query = state.get("query", "")

    system = SystemMessage(content=(
        "You are the Support Supervisor at ABC Technologies. Review the draft "
        "response below that a support agent has written for a customer. "
        "Improve it if needed for clarity, tone, and professionalism, but keep "
        "all factual details unchanged. Return ONLY the final polished response "
        "to send to the customer, with no extra commentary."
    ))
    human = HumanMessage(content=(
        f"Customer question: {query}\n\nDraft response:\n{draft}"
    ))

    response = llm.invoke([system, human])

    return {"final_response": response.content.strip()}


def make_memory_recall_node(checkpointer):
    """Factory that binds a memory_recall node to a specific checkpointer
    instance (the one opened against memory.db in graph.py)."""

    def memory_recall_node(state):
        customer_id = state.get("customer_id", "unknown")
        history = get_conversation_history(checkpointer, customer_id)

        if not history:
            answer = "I don't see any previous support issues on file for you yet."
        else:
            lines = []
            for h in history:
                lines.append(f"- ({h['intent']}) \"{h['query']}\" -> {h['final_response']}")
            history_text = "\n".join(lines)

            system = SystemMessage(content=(
                "You are a customer support assistant. Below is the customer's "
                "past support history. Answer their question about their "
                "previous issue(s) using only this history, in a friendly, "
                "concise way."
            ))
            human = HumanMessage(content=(
                f"Past support history:\n{history_text}\n\n"
                f"Customer's question: {state['query']}"
            ))
            response = llm.invoke([system, human])
            answer = response.content.strip()

        return {
            "department_response": answer,
            "final_response": answer,
            "requires_approval": False,
            "approval_status": "not_required",
        }

    return memory_recall_node
