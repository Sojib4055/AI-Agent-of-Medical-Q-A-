from src.agents.db_agents import get_heart_sql_agent


def query_heart_disease(question: str) -> str:
    agent = get_heart_sql_agent()

    try:
        result = agent.invoke({"input": question})
    except Exception as e:
        return (
            "I tried to query the heart disease dataset but ran into an internal error: "
            f"{e}. Please try rephrasing your question."
        )

    if isinstance(result, dict) and "output" in result:
        text = result["output"]
    else:
        text = str(result)

    lowered = text.lower()
    if "max iterations" in lowered or "iteration limit" in lowered:
        return (
            "I tried many SQL steps on the heart disease dataset but could not safely "
            "complete an answer. Please try asking more directly, for example:\n"
            "\"What is the maximum age in the heart_disease table?\""
        )

    return text
