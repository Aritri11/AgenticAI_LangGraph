What LangSmith is ?


LangSmith is presented as a unified observability and evaluation platform to debug, test, and monitor AI app performance.


It traces:


Application‑level inputs and outputs.


Intermediate steps (chains, retrievers, tools, parsers, etc.).


Latencies at app and component level.


Token usage and estimated cost.


Errors and exceptions.


Tags, metadata, and optional user feedback.


Why LangSmith? (three real‑world scenarios)


The video uses three detailed scenarios to motivate observability.

2.1 Latency debugging in a job‑assistant LLM app


App: LLM tool that reads job descriptions (JD), fetches a student’s documents from Google Drive, matches skills, and generates customized cover letters multiple times per day.


Problem: latency suddenly jumps from ~2 minutes to 7–10 minutes; users start complaining and churn.


Pain point: you only see input, final output, and total time; you cannot see step‑wise timings (JD analysis, portfolio fetch, matching, proofreading) to find the bottleneck.


Example root cause: code change causing the system to scan the entire Google Drive instead of a specific folder, massively increasing one internal step’s latency.


2.2 Cost debugging in an autonomous research agent


App: autonomous research assistant agent that fetches academic papers from Google Scholar/arXiv for a topic, extracts key points, summarizes them into a report, and lets users chat with the report.


Normal behavior: typical report costs about ₹0.50 in API tokens.


Problem: some reports suddenly cost ₹2 instead of ₹0.50, while others remain cheap, causing unpredictable spikes and loss of profitability at scale.


Likely cause: a small prompt change like “keep refining until you get a perfect report” turns the agent into a loop—re‑fetching papers, re‑reading, and re‑summarizing for certain topics, multiplying token usage.


Pain point: there’s no obvious error trace; behavior is non‑deterministic and only visible as cost spikes in the dashboard.


2.3 Quality / hallucination debugging in a RAG HR chatbot


App: internal RAG chatbot for a large company (e.g., TCS) that answers questions from HR policy documents (leave policy, notice period, health insurance, etc.) to help freshers.


Problem: chatbot starts hallucinating critical answers (e.g., “take leave whenever you want”), spreading misinformation.


Two typical failure sources in RAG:


Retriever errors: wrong or irrelevant chunks retrieved (e.g., company history instead of leave policy).


Generator errors: LLM ignores context or is poorly prompted, hallucinating answers.


Pain point: in production you only see the final answer; you cannot see what documents the retriever returned or how the final prompt looked, so you can’t tell if the issue is in retrieval or generation.


Conclusion of this section: LLM systems are complex, non‑deterministic, and hard to debug from just input/output and logs; we need a tool to “open the black box” and inspect internal steps.



Observability definition:


Observability is defined as the ability to understand a system’s internal state by examining external outputs like logs, metrics, and traces.


Goals: diagnose issues, understand performance, and improve reliability by analyzing system‑generated data; effectively, answer “why is this happening?” even for unanticipated problems.


LLM‑specific difficulty: behavior is non‑deterministic, there are multiple interacting components (retrievers, tools, chains), and traditional error traces are often absent.


LangSmith feature overview 


Key features of LangSmith include:


Tracing with hierarchical view (project → traces → runs) and rich UI.


Token and cost tracking per model and per execution.


Latency analysis at both overall and component granularities.


Error inspection and stack traces.


Tagging and metadata for organizing experiments and environments.


Test suites and evaluation support (comparing runs, collecting human feedback).


Integrations with LangChain and LangGraph via environment variables and lightweight code changes.


Other use cases of LangSmith:


1. Evaluation of LLM outputs and agents


LangSmith has a full evaluation framework with datasets and evaluators for scoring outputs, not just inspecting traces.


Evaluators include:


Heuristic checks (regex/structure validation, code‑compiles, JSON validity).


LLM‑as‑judge evaluators with custom criteria (helpfulness, faithfulness, toxicity, etc.).


Pairwise comparison evaluators (prompt A vs prompt B, model A vs model B).


Custom Python/TypeScript evaluators for domain‑specific metrics (e.g., correctness vs golden answers, guardrail checks).


2. Building and managing evaluation datasets


You can construct datasets of test inputs + reference outputs (gold labels) to benchmark apps.


Datasets can be built from:


Saved traces in debugging or production (turn “interesting failures/successes” into tests).


Manually curated or imported examples.
​

LangSmith maintains versioned datasets with audit trail, so you can pin a specific version for a given experiment/regression test and avoid dataset drift.


3. Experimentation and prompt/model selection


Comparison dashboards let you run the same dataset across different prompts, models, or architectures and view metrics side‑by‑side to choose the best trade‑off between quality, latency, and cost.
​
​
Pre‑built evaluators for common tasks (e.g., QA, RAG, generic criteria‑based grading) make it faster to explore design choices without writing your own metrics from scratch.


4. Multi‑turn and conversation‑level evaluation


LangSmith supports multi‑turn evals, treating entire conversations as “threads” and scoring them on:


Whether user intent was correctly understood.


Whether the overall task was completed.


How the agent’s tool‑use trajectory unfolded.
​

This is used for chatbots/agents where quality must be judged at dialog level, not just per single response.



5. Production monitoring and quality ops


While rooted in observability, LangSmith also supports ongoing quality operations:


Tracking long‑term quality metrics and regressions.


Combining traces, eval scores, and human labels to monitor “health” of the app over time.


Identifying segments (by tags/metadata) where performance is weak and focusing improvement there.
