

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = "memory.db"


def get_checkpointer():
    """Open (or create) memory.db and return a SqliteSaver checkpointer
    connected to it. check_same_thread=False because the connection gets
    reused across the interrupt/resume cycle in main.py."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


def get_conversation_history(checkpointer, customer_id: str):
    """Walk every checkpoint stored for a given customer and rebuild a
    clean, de-duplicated list of {intent, query, final_response} turns,
    oldest first.

    LangGraph writes several intermediate checkpoints per turn (one per
    node), so we replay them in order and just keep the latest values
    seen for each distinct query.
    """
    config = {"configurable": {"thread_id": customer_id}}

    # .list() comes back newest-first, so reverse it to replay in order
    checkpoints = list(checkpointer.list(config))

    turns = {}    # query -> {"intent": ..., "final_response": ...}
    order = []     # first-seen order of each distinct query

    for cp in reversed(checkpoints):
        values = cp.checkpoint.get("channel_values", {})
        query = values.get("query")
        intent = values.get("intent")
        final_response = values.get("final_response")

        if not query:
            continue

        if query not in turns:
            order.append(query)
            turns[query] = {"intent": None, "final_response": None}

        if intent:
            turns[query]["intent"] = intent
        if final_response:
            turns[query]["final_response"] = final_response

    history = []
    for q in order:
        entry = turns[q]
        if entry["final_response"]:  # only turns that actually completed
            history.append({
                "query": q,
                "intent": entry["intent"],
                "final_response": entry["final_response"],
            })

    return history
