What is persistence?


• Informal definition:


• Persistence in LangGraph is the ability to save and restore the state of a workflow over time.


• Default behavior without persistence:


• When a workflow run finishes, its state (including all intermediate values) exists only in memory (RAM) and is lost afterward.


• You cannot later inspect or reuse that state.


• With persistence:


• State snapshots are saved somewhere durable (typically a database), so you can later restore and reuse them.


Speciality of persistence (intermediate states)


• Persistence stores not only the final state of a workflow, but all intermediate states at each checkpoint.


• Example:


• State has key name.


• Initial value at start: A.


• Node 1 changes it to B.


• Node 2 changes it to C at the end.


• Persistence records:


• name = A before Node 1.


• name = B before Node 2.


• name = C at the end.


• This allows:


• Recovering any intermediate state.


• Resuming execution from where a crash happened (fault tolerance).


• “Time travel” and debugging by replaying from a chosen checkpoint.


Why persistence is crucial for chatbots


• Behavior of real chatbots (e.g., ChatGPT):


• Option to start a new conversation.


• Option to resume a previous conversation from days ago.


• To implement “resume chat”:


• You must store all messages (conversation history) that were in the workflow’s state at the time of earlier runs.


• That storage is done via persistence into a database.


• Without persistence:


• Past conversations are not available; you cannot resume, and you can’t show old chats.


Where/what is actually stored (databases)


• “Somewhere” you store state = typically a database.


• Persistence design:


• Every time a checkpoint is reached, the current state snapshot is written to the chosen storage backend.


• Later you can load final or intermediate states by identifiers.


• Benefits summarized:


• Fault tolerance.


• Resume chat / short‑term memory.


• Other advanced features like time travel and human‑in‑the‑loop.


Checkpointer: how persistence is implemented


• Persistence in LangGraph is implemented via a checkpointer.


• Role of checkpointer:


• Break workflow execution into checkpoints.


• At each checkpoint, save the current state (intermediate + final) into storage.


• How checkpoints are defined:


• Each superstep in the graph corresponds to a checkpoint.


• Superstep refresher:


• A superstep is a “layer” of execution; possibly multiple nodes executed in parallel count as a single superstep.


• Example graph:


• Superstep 1: Start → Node 1.


• Superstep 2: Node 1 → Nodes 2, 3, 4 (in parallel).


• Superstep 3: Nodes 2/3/4 → End.


• Checkpoints in that example:


• One before/after superstep 1, another around superstep 2, another around superstep 3, etc.


Numeric example with list reducer


• State has key numbers, a list of integers, with a reducer function that merges new values into the list.


• Run:


• Start: numbers = [1] (checkpoint 1 saved).


• Node 1 adds 2 → numbers = [1, 2] (checkpoint 2 saved).


• Nodes 2,3,4 generate 3,4,5 → final numbers = [1, 2, 3, 4, 5] (checkpoint 3 & end saved).


• Database ends up storing four state snapshots (for four checkpoints).


Threads: distinguishing different executions


• Problem: many runs of the same graph share the same database; how to know which states belong to which run?


• Solution: threads and thread_id.


• Concept:


• Each workflow run is associated with a thread_id.


• All state snapshots from that run are saved against that thread.


• Example:


• First run with initial numbers = [1] uses thread_id = 1. All checkpoints for this run are stored under thread 1.


• Second run with initial numbers = [6] uses thread_id = 2. All its checkpoints are stored under thread 2.


• Retrieval:


• To resume or inspect a specific run, query the database by thread_id (e.g., “give me all states for thread 2”).


• Chatbot analogy:


• Each conversation session is a separate thread; each session’s messages are saved under its own thread_id and can later be resumed.


Implementation in code: simple joke workflow


Setup
• Imports:
• LangGraph components (graph builder, etc.).
• InMemorySaver from langgraph.checkpoint.memory as the checkpointer.
• Note on InMemorySaver:
• Stores all state snapshots in RAM only.
• Good for demos and learning.
• In production, you use other checkpointers (Postgres, Redis, etc.) to persist to real databases.
State and graph definition
• State schema with keys (all strings):
• topic – topic of the joke.
• joke – generated joke text.
• explanation – explanation of the joke.
• Two node functions:
• generate_joke(state)
• Builds prompt: “Generate a joke on the topic: {topic}”.
• Calls LLM, writes result into state["joke"].
• generate_explanation(state)
• Prompt: “Write an explanation for the joke: {joke}”.
• Calls LLM, writes result into state["explanation"].
• Graph edges:
• START → generate_joke → generate_explanation → END.
Attaching the checkpointer
• Instantiate checkpointer:
• checkpointer = InMemorySaver().
• During graph.compile(...), pass this checkpointer so LangGraph knows to persist state at every checkpoint.
Running the workflow with persistence
• Invocation:
• workflow.invoke({"topic": "pizza"}, config={"configurable": {"thread_id": "1"}}).
• thread_id usage:
• All state snapshots for this run (with topic pizza) will be tagged with thread 1.
• After run:
• Use workflow.get_state(config={"configurable": {"thread_id": "1"}}) to fetch the final state (topic, joke, explanation).
• If using a real DB checkpointer and the app was closed, you could still fetch this final state later.
• Viewing intermediate state history:
• workflow.get_state_history(config={"configurable": {"thread_id": "1"}}) returns a list of snapshots.
• For each element you see:
• values (current state dict).
• next (which node will execute next).
• Example sequence:
• Before Start: empty state, next = Start.
• Before generate_joke: topic = "pizza", joke = None, explanation = None, next = generate_joke.
• Before generate_explanation: topic and joke filled, explanation still empty, next = generate_explanation.
• Before End: topic, joke, explanation all present, next = End / no next.
Second run with another topic
• Invoke with topic pasta and thread_id = "2".
• Now:
• get_state(thread_id=2) returns pasta joke and explanation.
• get_state(thread_id=1) still returns pizza joke and explanation, both stored separately.
• get_state_history for each thread gives their respective intermediate states, showing persistence across runs.


Benefits of persistence 


The video explicitly lists four main benefits of persistence in LangGraph.


1. Short‑term memory for chatbots.


2. Fault tolerance.


3. Human‑in‑the‑loop (HITL).


4. Time travel (replay and debugging).


Each is then explained with examples and demos.


1.Short‑term memory

• To provide “resume previous conversation” behavior, you need to:


• Store previous conversation states (messages, etc.) persistently.


• Associate them with a thread_id or conversation ID.


• Implementation details (in this video):


• The conceptual explanation refers back to another playlist video where a LangGraph chatbot with short‑term memory was built using persistence.


• Key point:


• In LangGraph, persistence is the only way to implement short‑term memory for chatbots because default state is ephemeral.


2.Fault tolerance (with live demo)


• Fault tolerance definition:


• If a workflow crashes mid‑execution (due to server down, API error, etc.), you can resume from where it crashed instead of restarting from the beginning.


• Demo workflow:


• Simple graph with nodes step1, step2, step3 and state keys: input, step1, step2, step3.


• Edges: Start → step1 → step2 → step3 → End.


• step2 artificially includes a 30‑second sleep to simulate a long‑running step.


• Procedure:


• Run workflow with thread_id = "1" and input = "start".


• Let step1 execute, then the run pauses in step2’s delay.


• Manually send a keyboard interrupt (simulate crash).


• Inspecting state after crash:


• get_state(thread_id=1) shows input="start", step1="done", no step2 or step3 yet, confirming crash occurred after step1.


• get_state_history(thread_id=1) shows the history up to that point (start → after step1).


• Resuming:


• To resume, call workflow.invoke(None, config={"configurable": {"thread_id": "1"}}).


• Passing None as state tells LangGraph to resume from last checkpoint, not start over.


• Execution continues from step2, then step3.


• Final get_state shows step1, step2, step3 all done.


• Takeaway:


• Fault tolerance uses persistence’s stored checkpoints to restart from the crash point.


3.Human‑in‑the‑loop (HITL) concept


• Use case example:


• Workflow:


• Take a topic.


• LLM generates a LinkedIn post.


• Then post it to LinkedIn via API.


• Requirement: before posting, ask a human for approval: “Should I post this or not?”


• Challenge:


• Human response might be immediate, or after an hour, or after 2 days.


• You cannot keep the workflow process alive in memory for such long durations.


• LangGraph approach:


• At the human‑approval stage, the workflow is temporarily interrupted/suspended.


• When the human response arrives, the workflow is resumed from that exact suspension point.


• Role of persistence:


• Since every checkpoint (including the HITL point) is stored, LangGraph knows where to resume once human input is present.


• Relationship to fault tolerance:


• Similar mechanism (resume from checkpoint), but HITL is a planned interruption waiting for human; fault tolerance is about handling unplanned crashes.


• Note:


• In this video, HITL is explained conceptually; a full implementation demo is deferred to a later dedicated video.


4.Time travel in LangGraph (with demo)


Concept


• Time travel = ability to:


• Jump back to a previous checkpoint of a workflow.


• Replay execution forward from that checkpoint, possibly with modified state.


• Main use: debugging complex workflows; re‑run parts where something went wrong or explore alternative paths.


Demo 1: replay from checkpoint


• Uses the same joke workflow as earlier (topic → joke → explanation).


• Setup:


• Already ran the graph for topic="pizza" and topic="pasta" with persistence, so multiple checkpoints exist.


• Steps:


5. Inspect get_state_history(thread_id="1") (pizza run) and find the checkpoint where:


• Topic is set ("pizza").


• Joke and explanation have not been generated yet.


6. Note the checkpoint_id for that specific snapshot (LangGraph attaches a checkpoint_id to each history record).


7. Call workflow.get_state(config={"configurable": {"thread_id": "1", "checkpoint_id": <that_id>}}) to load that intermediate state.


8. Then call workflow.invoke(None, config={"configurable": {"thread_id": "1", "checkpoint_id": <same_id>}}) to replay from there.


• This regenerates the joke and explanation from that point onward.


• Observation:


• Because the LLM is probabilistic, the new joke/explanation for pizza differs from the original run’s outputs.


• get_state_history now shows extra entries representing this new “branch” of execution (additional checkpoints).


Demo 2: editing state at a checkpoint before replay


• Goal: change the topic from “pizza” to “samosa” at a specific checkpoint and then replay downstream nodes.


• Steps:


10. Identify checkpoint where topic="pizza" and no joke yet (similar as before).


11. Use workflow.update_state(config={"configurable": {"thread_id": "1", "checkpoint_id": <that_id>}}, values={"topic": "samosa"}) to modify the stored state at that checkpoint.


12. This creates another branch in the state history where the topic is now samosa.


13. Find the new checkpoint id of this updated state in get_state_history.


14. Invoke the workflow from this new checkpoint:


• workflow.invoke(None, config={"configurable": {"thread_id": "1", "checkpoint_id": <new_id>}}).


15. Now the graph runs forward from that edited state, generating a joke and explanation about samosa.


• Correction highlighted in video:


• Initially, he mistakenly re‑invoked from the old pizza checkpoint and still got pizza jokes.


• Fix: you must re‑invoke from the new checkpoint created after update_state, not the original one.


• History view:


• First four entries: original “pizza” execution.


• Next two: first time‑travel replay.


• Another entry: state update to samosa.


• Final entry: replay from samosa checkpoint, generating samosa joke and explanation.


Practical note


• Time travel is mostly a debugging/development feature, especially useful for complex long‑running graphs.


• For simple flows you may rarely need it, but it shows the power of having full state history saved by persistence.


Persistence = Memory Across Runs

Persistence means saving the graph state so it doesn’t forget.

Without persistence:

When the program stops → everything is lost.

With persistence:

The state is stored (like in a database or file)

You can resume from where you left off


Reducers = How State Updates Inside One Run

Reducers control how multiple updates to the same state key are combined.

They work within a single graph execution.


Imagine writing notes in a notebook.

Reducer = How you combine notes written by multiple people on the same page.

Persistence = Putting the notebook in your bag so you still have it tomorrow.
