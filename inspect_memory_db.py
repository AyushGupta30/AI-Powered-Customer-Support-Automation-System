

import sqlite3

from memory_db import get_checkpointer, get_conversation_history

DB_PATH = "memory.db"


def show_schema():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    print("=== TABLES IN memory.db ===")
    for name, sql in cur.fetchall():
        print(f"\n-- {name} --")
        print(sql)
    conn.close()


def show_thread_ids():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT thread_id FROM checkpoints;")
    ids = [r[0] for r in cur.fetchall()]
    conn.close()
    print("\n=== CUSTOMER IDS (thread_id) WITH STORED MEMORY ===")
    for i in ids:
        print(" -", i)
    return ids


def show_history_for(customer_id):
    checkpointer = get_checkpointer()
    history = get_conversation_history(checkpointer, customer_id)
    print(f"\n=== CONVERSATION HISTORY for '{customer_id}' ===")
    if not history:
        print("(no history found)")
    for h in history:
        print(f"\nQ: {h['query']}")
        print(f"Intent: {h['intent']}")
        print(f"A: {h['final_response']}")


if __name__ == "__main__":
    show_schema()
    ids = show_thread_ids()
    for cid in ids:
        show_history_for(cid)
