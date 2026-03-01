
The need for memory and the deadlock


• Argues that almost every GenAI/agentic app needs memory: otherwise every user query is treated in isolation, leading to poor UX (repeating context every time).

​
• Identifies a “deadlock”:


• Fact 1: LLMs are stateless and have no built‑in memory.


• Fact 2: Useful GenAI apps require memory.


• Therefore, memory must be implemented externally, around the LLM


Context window


• Introduces context window: the maximum amount of text an LLM can read and “hold” at one time before answering.

​
• Uses a camera analogy: model is the camera, context window is the lens; bigger lens → more of the scene captured at once.

​
• Notes typical modern context windows (e.g., 128k tokens, some models up to 1M), translating roughly to hundreds of pages of text.​


In‑context learning


• Explains LLM training as acquiring parametric knowledge stored in \theta, learned from large corpora.

​
• Describes in‑context learning as an emergent ability: the model can also use information and patterns present inside the prompt itself (e.g., a private 100‑page PDF pasted in the prompt) to answer questions, even if that content was never in training data.

​
• Defines in‑context learning as using prompt information in addition to parametric knowledge to answer a query.​


Short‑term memory in chatbots (conversation‑scoped)


• Maps this idea to production chatbots like ChatGPT/Gemini: each conversation/thread has its own short‑term memory (the per‑thread conversation buffer).

​
• When you start a new conversation, the buffer is reset; STM is thread‑scoped—it exists only within a single session and does not cross conversations.

​
• Highlights why you don’t keep a single global buffer for all chats: it would be too long, incoherent, and hard for the model to use.

​
Limitations of short‑term memory


1 Fragility and lack of persistence


• STM lives in an in‑memory variable (e.g., messages); if the process restarts, server crashes, or you start “New chat”, the entire buffer is lost.

​
• Introduces persistence: storing the conversation buffer into a database keyed by thread ID before resetting, and re‑loading it when the user reopens that thread.

​
2 Context window limits


• As conversations grow long, concatenated history may exceed the context window, leading to incoherent responses or hallucinations.

​
• Presents common mitigation strategies:


• Trimming: keep only last N messages.


• Summarization + recency: summarize older parts of the conversation with another LLM, keep that summary plus the last K messages as input.

​
3 Thread‑scope and lack of cross‑conversation continuity


• Since STM is thread‑scoped, the model can’t remember user preferences across sessions (e.g., always prefer Python, travel budget, explanation style).

​
• Learning does not compound over time; the user has to re‑teach preferences in every new conversation.

​
• Cross‑thread reasoning is impossible: you can’t ask “What did we do last time?” or “What solution worked last week?” across different threads because STM does not span them.


Motivation for long‑term memory (LTM)


• To support personalization and continuity across sessions, the system needs a different kind of memory that:


• Stores special, long‑lasting information that survives beyond a single conversation (e.g., “user prefers Python”, “user is writing a book”).


• Is selective: only stable, useful, reusable information is kept, not the entire raw chat.

​
• Defines this as long‑term memory: information living outside any specific conversation, over long periods (days/months).​


 Types of long‑term memory

​
1. Episodic memory


• Stores what happened in past sessions: which solutions worked/failed, which deployments had wrong credentials, which actions were taken.​


• Enables questions like “What did we do last time?” or “Did we already try this approach?”.​


2. Semantic memory


• Stores facts about the user, system, and tasks: user prefers Python, user is a beginner, system uses Postgres, budget is ₹10,000, etc.​


• This is the main backbone for personalization and system‑level knowledge.​


3. Procedural memory


• Stores “how‑to” knowledge: strategies, rules, and learned behaviors (e.g., avoid subqueries, prefer window functions, always explain step‑by‑step).​


• Over time, this makes an agent feel more adapted and comfortable to the user’s style.​

 High‑level architecture of long‑term memory


The LTM system is described as a four‑step loop: creation, storage, retrieval, injection.​


4. Creation (or update)


• During a conversation, the system monitors user messages, model responses, and tool outputs to detect memory candidates.​


• Pipeline: extract candidates → filter noise → decide scope (user/app/agent level) → choose to create new memory, update existing, or ignore.​


5. Storage


• Persist selected memories into durable stores (so they survive restarts), attaching identifiers and metadata for later search.​


• Possible stores: relational DB, key‑value store, logs/text files, vector DB for semantic search—choice depends on memory type (episodic/semantic/procedural).​


6. Retrieval


• In a new conversation, before responding, the system asks: “Given this situation, what should I remember right now?”.​


• Uses the current user input to query the memory store and bring back a small, relevant subset of memories, not everything (retrieval is selective, not exhaustive).​


7. Injection


• Retrieved memories are not fed directly into the model as a separate channel; they are injected into short‑term memory (conversation buffer/context window) and become part of the prompt.​


• The model just sees more input tokens; long‑term memory always flows through short‑term memory into the LLM.​

Challenges in building memory systems


• Memory creation decision: deciding what is worth turning into long‑term memory from noisy chat streams is difficult.​


• Relevant retrieval in real time: for each turn, determining which subset of potentially large memory to fetch so that it actually helps the current query.​


• System orchestration: integrating memory creation, storage, retrieval, and injection with an already complex agentic system and its data stores/tools is non‑trivial engineering.​

Tools and platforms for memory


• Mentions emerging libraries and managed services that implement memory layers for GenAI apps, such as LangMem (from the LangChain ecosystem), Mem0, and Supermemory, which abstract much of the long‑term memory plumbing.​


• Emphasizes that this “memory around LLMs” space is rapidly growing, with active research and new products.​
