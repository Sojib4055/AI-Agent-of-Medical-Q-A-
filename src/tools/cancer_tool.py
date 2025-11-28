from src.agents.db_agents import get_cancer_sql_agent


def query_cancer_data(question: str) -> str:
    """
    Answer a question using the cancer.db dataset.
    """
    agent = get_cancer_sql_agent()

    try:
        result = agent.invoke({"input": question})
    except Exception as e:
        # Fallback if the agent crashes completely
        return (
            "I tried to query the cancer dataset but ran into an internal error: "
            f"{e}. Please try rephrasing your question."
        )

    # Normal AgentExecutor returns a dict like {"input": ..., "output": "..."}
    if isinstance(result, dict) and "output" in result:
        text = result["output"]
    else:
        text = str(result)

    # Clean up the ugly LangChain error if it appears
    lowered = text.lower()
    if "max iterations" in lowered or "iteration limit" in lowered:
        return (
            "I tried many SQL steps on the cancer dataset but could not safely "
            "complete an answer. Please try asking more directly, for example:\n"
            "\"What is the maximum Age value in the cancer_data table?\""
        )

    return text
