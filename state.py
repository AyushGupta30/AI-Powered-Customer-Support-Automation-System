

from typing import TypedDict, Optional


class SupportState(TypedDict, total=False):
    # customer info
    customer_id: str      # used as the memory thread_id (e.g. "cust_david")
    customer_name: str

    # incoming query
    query: str
    intent: str            # Sales | Technical | Billing | Account | Memory

    # RAG
    retrieved_context: str  # chunks pulled from the knowledge base for this query

    # department response
    department_response: str  # draft written by the specialist agent

    # human-in-the-loop
    requires_approval: bool
    approval_status: str        # not_required / pending / approved / rejected
    approval_notes: Optional[str]

    # what actually goes back to the customer
    final_response: str
