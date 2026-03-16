# CRAG: Corrective Retrieval-Augmented Generation

An implementation of **Corrective RAG (CRAG)** - a robust RAG architecture that eliminates blind trust in retrieved documents by evaluating their quality and conditionally routing to internal knowledge, external web search, or merged knowledge sources.

## 🎯 Traditional RAG & Failure Modes

**Standard RAG Pipeline:**
User Query → Embedding → Vector DB Retrieval → Retrieved Chunks + Query → LLM → Answer



**Demonstrated with LangGraph over 3 classic ML books:**
- Document loading → Recursive text splitting → OpenAI embeddings → FAISS vector store → Top-k retrieval

**Failure Cases:**
- ✅ **In-distribution**: "What is bias-variance tradeoff?" → Highly relevant chunks → Correct answer
- ❌ **Out-of-distribution**: 
  - Recent AI news → "I don't know" (no parametric knowledge)
  - Transformer architecture → Answers from params despite irrelevant chunks (hallucination)

## 🆕 CRAG Concept (Original Paper)

**Core Innovation**: Evaluate retrieved documents → Route based on quality:
Query → Retrieval → Evaluator → {Correct, Incorrect, Ambiguous} → Specialized Knowledge Path → LLM



**Three Retrieval Cases:**
| Case | Retrieved Docs | Action |
|------|----------------|--------|
| **Correct** | ≥1 strongly relevant | Refine internal knowledge |
| **Incorrect** | All broadly irrelevant | Web search (external) |
| **Ambiguous** | Somewhat relevant but incomplete | Merge internal + external |

## 📁 Step-by-Step Implementation

### 1. `2_knowledge_refinement.py`
**Knowledge Refinement** (for good retrievals):
Decomposition → Filtration → Recomposition


- Split chunks into sentence strips
- LLM judges each strip's relevance to query  
- Merge relevant strips into refined context

### 2. `3_thresholded_knowledge_refinement.py`
**Retrieval Evaluator** with Pydantic scoring (0-10 scale):


lower_threshold=3, upper_threshold=7

Correct: ≥1 doc > 7
Incorrect: All docs ≤ 3
Ambiguous: 3 < docs ≤ 7


Only docs > lower_threshold proceed to refinement.

### 3. `4_web_search_refinement.py`
**Incorrect Case**: Replace "fail" node with **Tavily web search**
> "Never leave users empty-handed - always provide high-quality answers"

### 4. `5_query_rewrite.py`  
**Query Optimization** for better web search:
"recent AI news" → "latest AI breakthroughs 2026 news articles"


Addresses vagueness, temporal constraints, keyword optimization.

### 5. `6_ambiguous_included.py`
**Ambiguous Case**: 
Keep good internal docs + Web search → Merge → Refine → Generate



## Implementation here vs Original CRAG Paper

**✅ Matches Paper Architecture:**
Retrieval → Evaluator → Internal/Web/Combined Paths



**⚠️ Implementation Differences:**
- **Ollama Chat LLM** instead of fine-tuned T5-Large (no released weights)
- **Heuristic thresholds** (3/7) - tunable per domain
- **LangGraph state management** for complex routing

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run step-by-step
python 2_knowledge_refinement.py
python 3_thresholded_knowledge_refinement.py
# ... continue through 6

# 3. Full CRAG pipeline
python 6_ambiguous_included.py

```
📚 Tech Stack
LangGraph (routing) + LangChain (RAG components)
OpenAI Embeddings + FAISS
Ollama (evaluator LLM) + Tavily (web search)
Pydantic (structured scoring)



