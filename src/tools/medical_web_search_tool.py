from typing import Optional

from langchain_groq import ChatGroq
from tavily import TavilyClient

from src.config import GROQ_API_KEY, TAVILY_API_KEY, ROUTER_MODEL


def _get_tavily_client() -> TavilyClient:
    """
    Create and return a Tavily client using the API key from .env.
    """
    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set in your .env file. "
            "Set it before using the MedicalWebSearchTool."
        )
    return TavilyClient(api_key=TAVILY_API_KEY)


def medical_web_search(question: str, *, max_results: int = 5) -> str:
    """
    MedicalWebSearchTool

    Use this tool ONLY for general medical knowledge questions such as:
      - definitions ("What is diabetes?")
      - symptoms ("What are the symptoms of heart disease?")
      - causes / risk factors
      - treatments, prevention, lifestyle advice
      - general medical explanations

    Do NOT use this tool for dataset-specific statistics, counts, or numeric analysis.
    For those, use the database tools instead (Heart, Cancer, Diabetes).
    """
    tavily = _get_tavily_client()

    # Step 1: Get search results from Tavily
    # NOTE: topic must be one of: "general", "news", "finance"
    # so we use "general" for medical info.
    search_result = tavily.search(
        query=question,
        search_depth="basic",
        max_results=max_results,
        topic="general",
    )

    # Tavily already returns some summary info. We'll extract relevant parts.
    # Structure typically: {"answer": "...", "results": [ ... ]}
    raw_answer: Optional[str] = search_result.get("answer")
    raw_results = search_result.get("results", [])

    # Build a context string from Tavily results
    context_chunks: list[str] = []
    if raw_answer:
        context_chunks.append(f"Tavily summary: {raw_answer}")
    for item in raw_results:
        title = item.get("title", "")
        content = item.get("content", "")
        context_chunks.append(f"{title}: {content}")

    context_text = "\n\n".join(context_chunks)

    # Step 2: Use a Groq-hosted LLM to produce a clear, short medical explanation
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set in your .env file. "
            "Set it before using the MedicalWebSearchTool."
        )

    llm = ChatGroq(
        model=ROUTER_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=180, 
    )

    prompt = f"""
You are a medical assistant. Using ONLY the information in the context below,
answer the user's medical question clearly and safely.

Rules for your answer:
- Be SHORT and DIRECT.
- Use at most 4 bullet points OR 3 short sentences.
- Start with the key number or definition if the question is about a range or normal value.
- If the topic relates to diagnosis or treatment, add ONE short sentence:
  "For personal medical advice, please consult a healthcare professional."

User question:
{question}

Context from web search:
{context_text}

Now give a brief answer.
"""


    response = llm.invoke(prompt)
    return response.content
