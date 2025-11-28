from functools import lru_cache

from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_groq import ChatGroq

from src.config import SQL_AGENT_MODEL, GROQ_API_KEY
from src.db import (
    get_heart_sql_database,
    get_cancer_sql_database,
    get_diabetes_sql_database,
)


def _make_llm() -> ChatGroq:
    """
    Create a Groq chat model instance for SQL agents.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set in your .env file. "
            "Set it before using the SQL database agents."
        )

    return ChatGroq(
        model=SQL_AGENT_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_heart_sql_agent():
    """
    LangChain SQL agent for the heart_disease.db database using Groq.
    """
    db = get_heart_sql_database()
    llm = _make_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="tool-calling",
        max_iterations=25,              # more room than 5
        early_stopping_method="generate",
    )
    return agent


@lru_cache(maxsize=1)
def get_cancer_sql_agent():
    """
    LangChain SQL agent for the cancer.db database using Groq.
    """
    db = get_cancer_sql_database()
    llm = _make_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="tool-calling",
        max_iterations=25,
        early_stopping_method="generate",
    )
    return agent


@lru_cache(maxsize=1)
def get_diabetes_sql_agent():
    """
    LangChain SQL agent for the diabetes.db database using Groq.
    """
    db = get_diabetes_sql_database()
    llm = _make_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="tool-calling",
        max_iterations=25,
        early_stopping_method="generate",
    )
    return agent
