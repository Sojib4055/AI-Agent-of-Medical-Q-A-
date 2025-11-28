import json
from dataclasses import dataclass
from typing import Literal

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, ROUTER_MODEL
from src.tools import (
    query_heart_disease,
    query_cancer_data,
    query_diabetes_data,
    medical_web_search,
)


ToolName = Literal["heart_db", "cancer_db", "diabetes_db", "web_search"]


@dataclass
class RoutingDecision:
    tool: ToolName
    query: str


ROUTER_SYSTEM_PROMPT = """
You are a routing assistant for a medical question-answering system.

You have access to four tools:

1) HeartDiseaseDBTool (tool name: "heart_db")
   - Use ONLY for questions that clearly ask about statistics, counts, or analysis
     that must come from the HEART DISEASE dataset.
   - Examples:
       - "How many patients are in the heart disease dataset?"
       - "What is the average age of patients with heart disease?"
       - "What percentage of patients have high cholesterol in this dataset?"

2) CancerDBTool (tool name: "cancer_db")
   - Use ONLY for questions that clearly ask about statistics, counts, or analysis
     that must come from the CANCER dataset.
   - Examples:
       - "How many cancer cases are in the dataset?"
       - "What proportion of tumors are malignant in this dataset?"

3) DiabetesDBTool (tool name: "diabetes_db")
   - Use ONLY for questions that clearly ask about statistics, counts, or analysis
     that must come from the DIABETES dataset.
   - Examples:
       - "How many patients have diabetes in this dataset?"
       - "What is the average BMI of diabetic patients vs non-diabetic patients?"

4) MedicalWebSearchTool (tool name: "web_search")
   - Use for GENERAL MEDICAL KNOWLEDGE, such as:
       - definitions ("What is diabetes?")
       - symptoms ("What are the symptoms of heart disease?")
       - causes and risk factors
       - treatments, medications, lifestyle advice
       - general pathophysiology or explanations
   - DO NOT use this for questions that clearly refer to "this dataset", "in this data",
     or ask for numeric statistics like counts, averages, percentages from the datasets.

Routing rules:

- If the user is clearly asking about numbers, statistics, percentages, counts, averages,
  or correlations IN A DATASET, choose one of the DB tools.
- If the user is asking about definition, symptoms, diagnosis, causes, treatments, prognosis,
  or general medical knowledge, choose "web_search".
- If the user mentions "dataset", "in this data", "in the heart dataset", etc.,
  strongly prefer the appropriate DB tool.
- If you are unsure, prefer "web_search".

Your task:

Given a single user question, decide:
  1) WHICH tool to use ("heart_db", "cancer_db", "diabetes_db", or "web_search")
  2) HOW to rewrite the question in a concise way for that tool as `query`.

You MUST respond in pure JSON with exactly this structure:

{
  "tool": "<one of: heart_db, cancer_db, diabetes_db, web_search>",
  "query": "<rewritten question optimized for that tool>"
}

No extra keys, no explanations, no markdown, ONLY valid JSON.
"""


def _get_router_llm() -> ChatGroq:
    """
    Create the LLM used for routing.
    """
    return ChatGroq(model=ROUTER_MODEL, api_key=GROQ_API_KEY, temperature=0)


def decide_tool(user_question: str) -> RoutingDecision:
    """
    Use the router LLM to decide which tool to call and how to phrase the query.
    """
    llm = _get_router_llm()

    # We send a system + user message and expect a pure JSON response.
    messages = [
        ("system", ROUTER_SYSTEM_PROMPT),
        ("user", user_question),
    ]

    response = llm.invoke(messages)
    raw_content = response.content

    # Try to parse JSON response
    try:
        data = json.loads(raw_content)
        tool = data.get("tool")
        query = data.get("query", user_question)
    except json.JSONDecodeError:
        # Fallback: simple keyword-based routing
        tool = _fallback_tool_choice(user_question)
        query = user_question

    # Final fallback validation
    if tool not in ("heart_db", "cancer_db", "diabetes_db", "web_search"):
        tool = _fallback_tool_choice(user_question)

    return RoutingDecision(tool=tool, query=query)


def _fallback_tool_choice(user_question: str) -> ToolName:
    """
    Simple heuristic routing if JSON parsing fails or tool is invalid.
    """
    q_lower = user_question.lower()

    # If clearly about heart dataset
    if "heart" in q_lower and ("dataset" in q_lower or "data" in q_lower):
        return "heart_db"

    # If clearly about cancer dataset
    if "cancer" in q_lower and ("dataset" in q_lower or "data" in q_lower):
        return "cancer_db"

    # If clearly about diabetes dataset
    if "diabetes" in q_lower and ("dataset" in q_lower or "data" in q_lower):
        return "diabetes_db"

    # If keywords suggest medical info (not stats)
    medical_keywords = ["symptom", "cause", "risk factor", "treatment", "cure", "diagnosis"]
    if any(kw in q_lower for kw in medical_keywords):
        return "web_search"

    # Default to web_search for safety
    return "web_search"


def run_routed_tool(decision: RoutingDecision) -> str:
    """
    Call the appropriate underlying tool based on the routing decision.
    """
    tool = decision.tool
    query = decision.query

    if tool == "heart_db":
        return query_heart_disease(query)
    elif tool == "cancer_db":
        return query_cancer_data(query)
    elif tool == "diabetes_db":
        return query_diabetes_data(query)
    elif tool == "web_search":
        return medical_web_search(query)
    else:
        # This should never happen, but just in case:
        return (
            "I could not determine the correct tool to use for your question. "
            "Please try rephrasing your question."
        )


def ask_medical_agent(user_question: str) -> str:
    """
    Main entry point for the multi-tool medical agent.

    - Takes a natural language medical question from the user
    - Uses a Groq-hosted model to decide which tool to use
      (HeartDiseaseDBTool, CancerDBTool, DiabetesDBTool, or MedicalWebSearchTool)
    - Calls that tool and returns the final natural language answer.
    """
    decision = decide_tool(user_question)
    answer = run_routed_tool(decision)
    return answer
