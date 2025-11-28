# 🧠 Multi-Tool Medical AI Agent

> **FastAPI + LangChain + Groq + SQLite + Tavily**\
> Ask anything about three medical datasets (Heart, Cancer, Diabetes)
> **or** general medical knowledge -- the agent routes your question to
> the right tool automatically.

------------------------------------------------------------------------

## 🌟 Project Overview

This project implements a **Multi-Tool AI Agent** that can:

1.  🔢 Answer **data-driven questions** from three structured medical
    datasets:
    -   **Heart Disease dataset** (SQLite: `heart_disease.db`)
    -   **Cancer dataset** (SQLite: `cancer.db`)
    -   **Diabetes dataset** (SQLite: `diabetes.db`)
2.  🌍 Answer **general medical knowledge questions** using **web
    search**:
    -   Definitions
    -   Symptoms, causes, risk factors
    -   Lifestyle and prevention information
    -   Treatment overviews and guidelines
3.  🧠 Automatically **route each question** to the appropriate tool:
    -   Heart/Cancer/Diabetes **DB tools** for dataset-specific
        statistics
    -   **MedicalWebSearchTool** for medical knowledge

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   FastAPI + Uvicorn\
-   LangChain (Agents + SQL Toolkits)\
-   Groq LLaMA3 Models\
-   Tavily Web Search API\
-   SQLite databases\
-   Python 3.10

------------------------------------------------------------------------

## 🗂️ Project Structure

    multi_tool_med_agent/
    ├── README.md
    ├── requirements.txt
    ├── .env.example
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── db/
    ├── notebooks/
    └── src/
        ├── config/
        ├── data_prep/
        ├── db/
        ├── tools/
        ├── agents/
        └── api/

------------------------------------------------------------------------

## 📥 Setup Instructions

### 1. Create Environment

    conda create -n MultiAgent python=3.11 -y
    conda activate MultiAgent

### 2. Install Dependencies

    pip install -r requirements.txt

### 3. Configure `.env`

    GROQ_API_KEY=your_key
    TAVILY_API_KEY=your_key
    APP_ENV=dev

### 4. Build SQLite Databases

    python -m src.data_prep.csv_to_sqlite

### 5. Start Backend

    uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000

------------------------------------------------------------------------

## 📡 API Usage

### POST `/ask`

Input:

``` json
{ "question": "What is the highest age of cancer patients?" }
```

Output:

``` json
{
  "question": "...",
  "answer": "In the cancer dataset, the highest recorded age is 80."
}
```

------------------------------------------------------------------------

## 🧪 Example Questions

### Dataset Questions

-   "What is the average cholesterol level of male heart patients?"
-   "What is the highest age of cancer patients?"
-   "How many patients have diabetes (Outcome = 1)?"

### Web Search Questions

-   "What is heart disease?"
-   "What is a healthy BMI range?"
-   "What are symptoms of high blood sugar?"

------------------------------------------------------------------------

## 🔍 Internal Architecture

### Tools:

-   `HeartDiseaseDBTool`
-   `CancerDBTool`
-   `DiabetesDBTool`
-   `MedicalWebSearchTool`

### Agents:

-   SQL Agents (Groq + SQLDatabaseToolkit)
-   Main Routing Agent (Groq LLaMA3-70B)

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.\
It is **not** a medical device and should not be used for diagnosis or
treatment.

------------------------------------------------------------------------

## ✔️ Quick Start

    python -m src.data_prep.csv_to_sqlite
    uvicorn src.api.app:app --reload

Enjoy your Multi-Tool Medical AI Agent!
