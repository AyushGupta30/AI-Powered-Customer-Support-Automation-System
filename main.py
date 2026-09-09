from graph import build_graph

CUSTOMER_ID = "cust_david"
CUSTOMER_NAME = "David"

DEMO_QUERIES = [
    "What are the pricing plans available for your software?",
    "I forgot my account password.",
    "My application crashes whenever I upload a file.",
    "I need a refund for my annual subscription.",
    "What was my previous support issue?",
]


def ask_human_for_approval(state) -> tuple:
    """Stand-in for a human supervisor reviewing a high-risk request in
    the terminal. Returns (decision, notes)."""
    print("\n" + "!" * 70)
    print("HUMAN-IN-THE-LOOP APPROVAL REQUIRED")
    print("!" * 70)
    print(f"Customer query : {state['query']}")
    print(f"Intent         : {state['intent']}")
    print(f"Draft response : {state.get('department_response')}")
    print("-" * 70)

    while True:
        decision = input("Approve this request? (y/n): ").strip().lower()
        if decision in ("y", "yes"):
            return "approved", ""
        elif decision in ("n", "no"):
            notes = input("Optional note for the customer (or press Enter): ").strip()
            return "rejected", notes
        else:
            print("Please type 'y' or 'n'.")


def run_query(graph, customer_id: str, customer_name: str, query: str):
    config = {"configurable": {"thread_id": customer_id}}

    print("\n" + "=" * 70)
    print(f"CUSTOMER QUERY: {query}")
    print("=" * 70)

    result = graph.invoke(
        {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "query": query,
        },
        config,
    )

    # if the graph paused for approval, handle it before moving on
    state = graph.get_state(config)
    while state.next:  # non-empty means execution is paused before a node
        if "human_approval" in state.next:
            decision, notes = ask_human_for_approval(state.values)
            update = {"approval_status": decision}
            if notes:
                update["approval_notes"] = notes
            graph.update_state(config, update)
            result = graph.invoke(None, config)
            state = graph.get_state(config)
        else:
            break

    print("-" * 70)
    print(f"Intent classified as : {result.get('intent')}")
    print(f"Approval required?   : {result.get('requires_approval')}")
    print(f"FINAL RESPONSE:\n{result.get('final_response')}")

    return result


def main():
    graph, checkpointer = build_graph()

    for query in DEMO_QUERIES:
        run_query(graph, CUSTOMER_ID, CUSTOMER_NAME, query)

    print("\n" + "=" * 70)
    print("Demonstration complete. Conversation history is stored in memory.db")
    print("=" * 70)


if __name__ == "__main__":
    main()
