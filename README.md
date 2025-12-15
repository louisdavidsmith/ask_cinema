# Cinema Expert Agent

This repository contains a small experimental agent focused on cinema knowledge and film recommendation, along with a data-driven evaluation framework for measuring and improving its behavior over time.

The project explores how large language models can be combined with structured tools, retrieval systems, and traditional machine-learning metrics to build agents that are both useful and measurable.

---

## Overview

The system consists of:

* A multi-step LLM agent that reasons about user intent and selectively uses tools
* A set of domain-specific tools for retrieval, structured data, and web search
* An offline evaluation suite that measures agent performance using non-LLM metrics

Everything runs locally and is designed to be simple to inspect, modify, and extend.

---

## Running the Project

### Installation

To run the project you will need both an open-ai-api key and a tmdb-api-key. To get a tmdb api key please follow the instructions here: https://developer.themoviedb.org/docs/getting-started. 

```bash
make install
```

This will:

* Create a virtual environment
* Install dependencies
* Download MovieLens data
* Build a local DuckDB datastore with embeddings

Environment variables are loaded from a `.env` file and include API keys, model identifiers, and datastore paths. I have included an .env.example file with defaults.

---

### Running the Agent

To start the API server:

```bash
make run
```



By default the service runs on `http://localhost:8000`.

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"healthy"}
```

### Query the Cinema Expert Agent

Send a request to the agent endpoint:

```bash
curl -X POST http://localhost:8000/cinema-expert \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "I like films by Tarkovsky and Bergman. What should I watch next? My favorite Bergman film is The Seventh Seal"
  }'
```

This endpoint executes the full multi-step agent flow, including tool calls (local datastore, metadata lookup, and synthesis), and returns a structured JSON response. 

---

For interactive use, a simple Streamlit UI is available:

```bash
make chat
```

## Agent Design

The agent is implemented as a two-stage process:

1. **Intent interpretation and tool selection**
   The model receives a system prompt and user input and decides whether additional information is needed.

2. **Tool execution and response synthesis**
   Tool outputs are injected back into the context and used to generate a grounded final response.

This structure keeps tool usage explicit while allowing the model to focus on reasoning and explanation.

---

## Tools

### Semantic Movie Retrieval

A local vector-based retrieval system is used for recommendations:

* User intent is embedded with a sentence-transformer model
* Movies are retrieved using cosine similarity in DuckDB
* Bayesian-adjusted rating thresholds help balance popularity and recall

This provides a lightweight retrieval-augmented generation setup that runs entirely locally.

---

### Movie Metadata (TMDB)

Structured movie information (cast, crew, release data) is retrieved via the TMDB API and used to enrich factual responses.

---

### Web Search

For film criticism and historical context, the agent can perform constrained web searches over a small set of high-signal domains.

---

## Evaluation

The repository includes an evaluation script that measures different aspects of agent behavior using traditional metrics.

Evaluations can be run with:

```bash
make assess
```

or directly:

```bash
python assess_performance.py --all
```

---

## Metrics

The evaluation suite includes:

* **Domain knowledge accuracy** using a cinema theory test
* **Embedding similarity correlation** between user preference representations and held-out movie ratings
* **Binary taste classification accuracy** for preference prediction

These metrics make it possible to track changes over time as models, prompts, or tools are adjusted.

---

## Notes

This project is intentionally scoped as an experiment rather than a polished application. Several areas are left open for iteration, including tool abstractions, retrieval strategies, and prompt structure.

The emphasis is on clarity, inspectability, and measurable behavior rather than completeness.

---
