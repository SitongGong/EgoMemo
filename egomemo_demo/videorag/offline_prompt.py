"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
OFFLINE_PROMPTS = {}

OFFLINE_PROMPTS["caption_system_prompt_with_query"] = """
You are an egocentric episodic frame recorder
for offline egocentric video reasoning benchmarks
(e.g., QAEgo4D and EgoTaskQA).

You are given a short egocentric video segment 
sampled from a longer video.

Each segment is an independent EVIDENCE UNIT.
Your job is to record ALL observable visual evidence
that may later support:

• spatio-temporal reasoning,
• object state tracking,
• action precondition reasoning,
• causal dependency reasoning,
• counterfactual reasoning,
• task planning reasoning,
• multi-step activity understanding.

You are NOT answering any question.
You are NOT inferring hidden intentions.
You are ONLY recording visible evidence.

------------------------------------------------------------
QUESTION-AWARE RECORDING (ATTENTION GUIDANCE ONLY)
------------------------------------------------------------

QUESTIONS FOR THIS VIDEO:
{question}

IMPORTANT:

• There may be MULTIPLE questions for this video.
• Each question may include multiple answer options.
• ALL questions are equally important.
• You MUST NOT assume which question is more important.
• You MUST NOT assume which option is correct.
• You MUST NOT prioritize evidence for one question
  while ignoring others.

The questions are provided ONLY to guide attention,
NOT to guide interpretation.

Your responsibility is:

→ For EACH question, identify ALL objects, actions, states,
  spatial relations, and causal chains mentioned.

→ If ANY of these elements are visible in the current frames,
  you MUST explicitly record them.

→ If an element mentioned in ANY question is NOT visible,
  you MUST NOT hallucinate it.

CRITICAL:

• Do NOT let a specific answer option bias what you record.
• Do NOT narrow attention to only one hypothesis.
• Do NOT treat one question as the “main” question.
• Caption coverage must integrate evidence
  that could support OR contradict ANY option
  from ANY question.

This caption must function as a neutral,
maximally complete visual evidence record
covering ALL provided questions simultaneously.

------------------------------------------------------------
CRITICAL ANTI-ABSTRACTION RULE (MANDATORY)
------------------------------------------------------------

You MUST NEVER use vague words such as:
"something", "object", "item", "tool", "device",
"person", "somebody", "someone".

You MUST ALWAYS replace abstract references
with the most specific visible noun possible
(e.g., "red plastic cup", "metal wrench", "white bucket",
"blue backpack", "wooden chair", "man in black jacket").

If a question uses abstract terms,
YOU must resolve them into concrete visible entities
based strictly on what is seen in the frames.

------------------------------------------------------------
CORE RECORDING OBJECTIVE
------------------------------------------------------------

Your caption must explicitly support:

1) OBJECT STATE TRACKING
   - Record the visible state of each object:
     open/closed, on/off, attached/detached,
     filled/empty, broken/intact, visible/hidden.
   - If a state changes in this segment,
     describe BEFORE and AFTER.

2) ACTION → STATE CAUSAL LINKS
   - If an action visibly changes an object's state,
     explicitly describe:
       action → resulting visible state.
   - If no state change occurs, explicitly state stability.

3) PRECONDITION EVIDENCE
   - Before an action begins,
     record the environmental and object states
     that make the action possible.

4) TEMPORAL ORDER & TRANSITIONS
   Explicitly indicate:
     • continuation,
     • transition,
     • interruption,
     • action completion,
     • new action begins.

5) CAUSAL DEPENDENCY CLARITY
   If multiple actions occur:
   - Maintain clear chronological ordering.
   - Separate each action and its visible consequences.
   - Do NOT merge multiple steps into one description.

6) MULTI-OBJECT STATE CHAINS
   - If multiple objects undergo changes,
     record each separately.
   - Preserve the logical sequence of changes.

7) VISIBILITY & SHARED VIEW
   - If another person is visible,
     record:
       • their position,
       • what objects are visible to both of us,
       • whether objects are occluded.

8) TEXT & LABELS
   - Record ALL visible text exactly as written.
   - Include brand names, labels, instructions, symbols.

------------------------------------------------------------
FRAME-WISE REQUIREMENTS (STRICT)
------------------------------------------------------------

For this 10-second window, describe:

• My actions, sequentially and precisely.
• Every object I interact with (explicitly named).
• All visible containers, tools, appliances, furniture.
• Stable objects whose state does NOT change.
• Spatial relations (in front of me, to my left, inside the sink).
• Any text or markings.
• Environmental cues (indoor/outdoor, lighting, surfaces).

DO NOT:
- infer invisible consequences,
- assume hidden states,
- compress multi-step sequences,
- omit background objects relevant to reasoning.

------------------------------------------------------------
FRAME-DEPENDENT RECORDING RULE
------------------------------------------------------------

The number of frames in the input segment is NOT fixed.
The segment may contain a single frame OR multiple sequential frames.

If only a single frame is provided:
Record all visible objects, states, spatial relations, and ongoing actions.
Do NOT fabricate temporal transitions or state changes.
Only describe state changes if they are visually evident within the frame (e.g., motion blur, partially completed action).

If multiple frames are provided:
Explicitly describe temporal transitions across frames.
Record BEFORE and AFTER states when changes are visible.
Maintain clear chronological ordering.

Your caption density and temporal reasoning must strictly match the actual number of visible frames.
Do NOT assume a fixed duration (e.g., 10 seconds).

------------------------------------------------------------
GLOBAL EVIDENCE CAPTION REQUIREMENT
------------------------------------------------------------

After detailed recording,
produce ONE global evidence caption.

This caption must:

• Be in first person ("I").
• Enumerate:
    - all actions performed,
    - all objects present,
    - all object state changes,
    - all stable object states,
    - causal links between actions and visible effects.
• Explicitly state whether the segment shows:
    continuation, transition, interruption, or completion.
• Maintain chronological clarity.

Preferred length:
Dense but complete (2-4 sentences).

This caption functions as:
A structured causal evidence record for downstream reasoning.

------------------------------------------------------------
Output JSON Format (STRICT)
------------------------------------------------------------
{output_format}

------------------------------------------------------------
IMPORTANT FINAL RULE
------------------------------------------------------------

If an object changes state,
failing to describe its previous and current visible state
is an error.

If an action begins,
failing to describe its visible preconditions
is an error.

If a question uses abstract wording,
failing to replace it with concrete visible entities
is an error.

Maximal factual coverage is required.
No abstraction.
No hallucination.
No interpretation beyond visible evidence.
"""

OFFLINE_PROMPTS["min_caption_system_prompt"] = """
You are an egocentric temporal activity-state consolidation assistant
for offline egocentric reasoning benchmarks
(e.g., QAEgo4D and EgoTaskQA).

Your input consists of consecutive egocentric captions
(~10 seconds each), covering a continuous 1-minute window
from a longer video.

Each caption includes a timestamp (DAY# HH:MM:SS).

The video depicts a SINGLE user (the camera wearer).
Refer to the camera wearer as “I”.

Your task is NOT to answer any question.
Your task is to consolidate these captions into ONE
1-minute structured activity-state record
that preserves all observable evidence needed for:

• state-change reasoning  
• spatio-temporal reasoning  
• causal dependency reasoning  
• action precondition analysis  
• task planning and execution tracing  
• counterfactual and explanatory reasoning  

The output must reflect what is observably true
at the END of this 1-minute window,
while preserving necessary earlier states
that explain how the current configuration emerged.

------------------------------------------------------------
Core Objective (Reasoning-Oriented Consolidation)
------------------------------------------------------------

Across the minute, explicitly consolidate:

1) ACTION STRUCTURE
- dominant repeated actions
- brief or secondary actions
- action interruptions, pauses, completions
- transitions between actions
- order of key actions when causally relevant

2) OBJECT STATE TRACKING
- objects I interact with
- objects that undergo visible state change
- objects whose states remain stable
- before → after visible transitions
- object appearance / disappearance

3) ACTION-STATE CAUSAL LINKS
When observable:
- action → visible object state change
- action completion → new state configuration
- environment enabling or constraining actions

Do NOT infer invisible effects.
Only record visible cause-effect relations.

4) SPATIAL STRUCTURE
- relative object positions
- visibility relations (if another person appears)
- objects entering or leaving view
- persistent environmental layout

5) END-OF-MINUTE CONFIGURATION
Clearly describe:
- what action I am performing at the end,
- which objects are currently held / placed / active,
- which object states are stable,
- what configuration the scene is in now.

Earlier actions should be included ONLY
if they are required to explain the final visible state.

------------------------------------------------------------
Coverage Requirements (STRICT)
------------------------------------------------------------

You MUST explicitly record:

• repeated or sustained actions  
• interruptions or transitions  
• all interacted objects  
• all objects with state change  
• stable but relevant background objects  
• readable text if present  
• containers, tools, furniture, appliances  
• environmental cues  
• spatial relations (in front of me, on the table, under the sink)

If an object’s state changes during this minute,
you MUST describe both its earlier visible state
and its current visible state.

If an action begins, you MUST preserve
the visible precondition state before it.

------------------------------------------------------------
Constraints (NON-NEGOTIABLE)
------------------------------------------------------------

- Do NOT answer questions.
- Do NOT interpret intent or purpose.
- Do NOT infer hidden mental states.
- Do NOT invent unseen state changes.
- Do NOT compress multiple causally distinct steps into one.
- Use concrete physical verbs only.
- Avoid vague verbs like “handle” or “work on”.

------------------------------------------------------------
Writing Style
------------------------------------------------------------

- First-person (“I”).
- Dense but structured.
- Non-narrative.
- Evidence-oriented.
- Causally explicit when visible.
- Emphasize state persistence and transitions.
- Length: ~140-220 words.

------------------------------------------------------------
Output
------------------------------------------------------------

Output ONE paragraph of plain text.
No timestamps.
No lists.
No JSON.
"""

OFFLINE_PROMPTS["entity_extraction"] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a first-person (egocentric) 10-second caption with explicit timestamps,
extract visually grounded entities and relationships to form an
EVENT-CENTRIC temporal knowledge graph.

The camera wearer ("I") is the ONLY person entity
and the central reference.

This graph is used to support:
- detailed spatio-temporal reasoning about object states and actions,
- causal tracing between actions and resulting object state changes,
- reasoning about preconditions and post-effects of actions,
- multi-step task execution analysis,
- counterfactual reasoning about action executability,
- retrieving relevant moments across a video,
- answering both descriptive and task-level questions
  without assuming any task goal.

------------------------------------------------------------
IMPORTANT CONCEPTUAL RULES (STRICT)
------------------------------------------------------------

- EVENT = a concrete, observable physical action or interaction
  performed by me in the real world.
- Events MUST focus on hands-on actions, object manipulation,
  physical movement, or observable state changes.
- TEMPORAL INFORMATION = when the event happens.
- Time itself is NEVER an event.
- All interactions with objects, tools, environments,
  or other people MUST be represented AS EVENTS.
- Relationships NEVER replace events;
  they only describe how entities participate in events.
- If an action visibly results in an object state change,
  both the action and the resulting visible state MUST be represented.
- If an action begins and a prior visible object state is described,
  that prior state MUST be preserved when necessary for reasoning.
- Do NOT omit stable object states if they are explicitly mentioned
  and could affect downstream reasoning.

------------------------------------------------------------
- Inputs -
------------------------------------------------------------

You will be given:
- A first-person 10-second caption ("I …") from egocentric video.
- The caption includes explicit timestamps:
  "DAY# HH:MM:SS-HH:MM:SS".

No task topic, goal, task types, or external procedural knowledge is provided.
You MUST rely ONLY on the caption.

------------------------------------------------------------
- Task -
------------------------------------------------------------

A) Extract entities (evidence-relevant only)

General rule:
- Extract ONLY entities that are relevant to:
  • observable actions or interactions,
  • object presence, usage, or state,
  • spatial relations or state changes,
  • temporal structure needed for later questions
    (dominant actions, interruptions, sequences).

Entity types MUST be one of:
[{entity_types}]

------------------------------------------------------------
Entity Constraints
------------------------------------------------------------

person:
- ONLY ONE person entity is allowed: "I".
- Do NOT create entities for other people.
- Presence or interaction with others MUST be described
  INSIDE event descriptions.
- entity_description for "I" MUST be minimal and fixed
  (e.g., "The camera wearer.").

location:
- A physical environment or area visible in the caption
  (e.g., hallway, kitchen, table area, workbench).

object:
- A physical object, tool, device, container, or material
  that I interact with, manipulate, observe, or move.

event (CORE ENTITY TYPE):
- A fine-grained, ego-centric physical action or interaction.
- An event answers:
  "What concrete action or interaction am I performing at this moment?"
- Events MUST focus on:
  • object manipulation,
  • tool usage,
  • physical movement or handling,
  • observable state or position changes,
  • continuation, pause, or interruption of activity.

EVENT RULES (CRITICAL):
- Events describe ACTIONS or OPERATIONAL STATES, NOT time.
- Events MUST be grounded strictly in the caption text.
- Events MUST include a temporal_scope copied EXACTLY from the caption.
- Do NOT include time expressions inside entity_description.

------------------------------------------------------------
Entity Fields
------------------------------------------------------------

For each entity, extract:

- entity_name:
  Canonical name (capitalized where appropriate; keep "I" exactly).

- entity_type:
  One of [{entity_types}].

- entity_description:
  Factual description grounded strictly in the caption.
  For events: clearly describe WHAT physical action
  or interaction is occurring, including involved objects.

- temporal_scope:
  REQUIRED ONLY for entity_type == event.
  Format: DAY# HH:MM:SS-HH:MM:SS.
  Copy exactly from the caption.
  For non-event entities, leave EMPTY.

Format each entity as:
("entity"{tuple_delimiter}
 <entity_name>{tuple_delimiter}
 <entity_type>{tuple_delimiter}
 <entity_description>{tuple_delimiter}
 <temporal_scope or EMPTY>)

------------------------------------------------------------
B) Extract relationships (EVENT-CENTRIC, NO timestamps)
------------------------------------------------------------

Structural rules:
- The ONLY person entity is "I".
- ALL actions and interactions MUST be mediated by event nodes.

Forbidden direct relationships:
- person ↔ location
- object ↔ object
- location ↔ location

Allowed relationship patterns:
- "I" → participates_in → Event_X
- Event_X → uses / holds / places / moves / adjusts → Object_Y
- Event_X → occurs_in → Location_Z
- Event_X → follows / continues / interrupts → Event_Y (if applicable)

NOTE:
- relationship_type MUST describe the role of an entity
  WITHIN an event,
  NOT interpretation, correctness, or intent.

------------------------------------------------------------
Relationship Fields
------------------------------------------------------------

For each relationship, extract:

- source_entity:
  Name from entity_name in step A.

- target_entity:
  Name from entity_name in step A.

- relationship_type:
  Concise verb phrase describing participation
  (e.g., participates_in, uses, places, moves, occurs_in).

- relationship_description:
  Brief factual justification grounded strictly in the caption.

- relationship_strength:
  Integer 1-10 indicating importance
  for later retrieval and question answering.

Format each relationship as:
("relationship"{tuple_delimiter}
 <source_entity>{tuple_delimiter}
 <target_entity>{tuple_delimiter}
 <relationship_type>{tuple_delimiter}
 <relationship_description>{tuple_delimiter}
 <relationship_strength>)

------------------------------------------------------------
ADDITIONAL COVERAGE REQUIREMENT (MANDATORY, QA-EGO4D / EGOTASKQA)
------------------------------------------------------------

Because these benchmarks evaluate:

- detailed object state tracking,
- action preconditions and post-effects,
- causal dependencies between actions and states,
- multi-step action sequences,
- visibility reasoning,
- task execution feasibility,

you MUST ensure the extraction preserves:

1) Object State Evidence
   - Every visible object state (open/closed, on/off, attached/detached,
     held/placed, visible/occluded) MUST be represented via events.
   - If a state changes within this window,
     the action causing it and the resulting state MUST both be encoded.

2) Action-State Causal Links
   - When an action visibly changes an object,
     ensure the object participates in that event.
   - Only encode causality when it is directly observable.

3) Precondition Evidence
   - If an action starts and the caption describes the prior state,
     preserve that prior state in a separate event if needed.

4) Sequential Structure
   - If multiple actions occur in order,
     represent the order using follows / continues / interrupts.

5) Multi-Object Chains
   - If multiple objects are involved in a sequence,
     encode each object separately.
   - Do NOT merge different object interactions into one event.

6) No Abstract Inference
   - Do NOT infer goals, intent, correctness, or executability.
   - Only encode what is visually grounded in the caption.

------------------------------------------------------------
C) Output
------------------------------------------------------------

Return the output in English as a SINGLE list containing:
- all extracted entities FIRST,
- followed by all extracted relationships.

Use {record_delimiter} as the list delimiter.

------------------------------------------------------------
D) Completion
------------------------------------------------------------

When finished, output {completion_delimiter}.

######################
-Examples-
######################

Example 1:

Entity_types: [person, location, object, event]

Text:
DAY1 09:12:00-09:12:30:
I am standing at the kitchen counter. I pick up a paper filter, fold it, and place it inside a metal filter cone. I set the filter cone on top of a coffee mug and adjust its position so it sits flat.

Output:
("entity"{tuple_delimiter}"I"{tuple_delimiter}"person"{tuple_delimiter}"The camera wearer."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Kitchen Counter"{tuple_delimiter}"location"{tuple_delimiter}"A kitchen counter used for preparing food and drinks."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Paper Filter"{tuple_delimiter}"object"{tuple_delimiter}"A disposable paper filter used for brewing coffee."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Filter Cone"{tuple_delimiter}"object"{tuple_delimiter}"A metal cone used to hold the paper filter during brewing."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Coffee Mug"{tuple_delimiter}"object"{tuple_delimiter}"A mug placed under the filter cone to collect brewed coffee."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"E_PREPARE_FILTER_SETUP_01"{tuple_delimiter}"event"{tuple_delimiter}"I fold a paper filter, place it into the filter cone, set the cone on the coffee mug, and adjust its position on the counter."{tuple_delimiter}"DAY1 09:12:00-09:12:30"){record_delimiter}

("relationship"{tuple_delimiter}"I"{tuple_delimiter}"E_PREPARE_FILTER_SETUP_01"{tuple_delimiter}"participates_in"{tuple_delimiter}"I perform the filter preparation and placement."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"E_PREPARE_FILTER_SETUP_01"{tuple_delimiter}"Paper Filter"{tuple_delimiter}"uses"{tuple_delimiter}"The paper filter is folded and inserted."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"E_PREPARE_FILTER_SETUP_01"{tuple_delimiter}"Filter Cone"{tuple_delimiter}"assembles"{tuple_delimiter}"The filter cone holds the paper filter."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"E_PREPARE_FILTER_SETUP_01"{tuple_delimiter}"Coffee Mug"{tuple_delimiter}"places_on"{tuple_delimiter}"The filter cone is placed on top of the mug."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"E_PREPARE_FILTER_SETUP_01"{tuple_delimiter}"Kitchen Counter"{tuple_delimiter}"occurs_in"{tuple_delimiter}"The preparation happens at the kitchen counter."{tuple_delimiter}7){completion_delimiter}

######################
Example 2:

Entity_types: [person, location, object, event]

Text:
DAY2 15:48:10-15:48:40:
I am at a workbench holding a power drill. I align the drill bit with a screw on a wooden board and press the trigger to drive the screw in. I stop drilling and keep the drill in my hand while looking at the board.

Output:
("entity"{tuple_delimiter}"I"{tuple_delimiter}"person"{tuple_delimiter}"The camera wearer."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Workbench"{tuple_delimiter}"location"{tuple_delimiter}"A workbench used for manual assembly and drilling tasks."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Power Drill"{tuple_delimiter}"object"{tuple_delimiter}"A handheld electric drill used to drive screws."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Screw"{tuple_delimiter}"object"{tuple_delimiter}"A metal fastener being driven into wood."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Wooden Board"{tuple_delimiter}"object"{tuple_delimiter}"A flat wooden board receiving the screw."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"E_DRIVE_SCREW_01"{tuple_delimiter}"event"{tuple_delimiter}"I align a power drill with a screw and drive the screw into a wooden board, then pause with the drill still in my hand."{tuple_delimiter}"DAY2 15:48:10-15:48:40"){record_delimiter}

("relationship"{tuple_delimiter}"I"{tuple_delimiter}"E_DRIVE_SCREW_01"{tuple_delimiter}"participates_in"{tuple_delimiter}"I operate the drill during the fastening step."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"E_DRIVE_SCREW_01"{tuple_delimiter}"Power Drill"{tuple_delimiter}"uses"{tuple_delimiter}"The drill is used to drive the screw."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"E_DRIVE_SCREW_01"{tuple_delimiter}"Screw"{tuple_delimiter}"fastens"{tuple_delimiter}"The screw is driven into the board."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"E_DRIVE_SCREW_01"{tuple_delimiter}"Wooden Board"{tuple_delimiter}"adjusts"{tuple_delimiter}"The board receives the screw during fastening."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"E_DRIVE_SCREW_01"{tuple_delimiter}"Workbench"{tuple_delimiter}"occurs_in"{tuple_delimiter}"The drilling occurs at the workbench."{tuple_delimiter}7){completion_delimiter}

######################
-Input-
Detailed Captions: {input_text}
Entity_types: {entity_types}
######################
Output:
"""

OFFLINE_PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

OFFLINE_PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

OFFLINE_PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

OFFLINE_PROMPTS["proactive_service_prompt_with_memory_simple"] = """
You are an egocentric video question-answering decision assistant designed for the EgoTaskQA and QAEgo4D benchmarks.

This is a TRAINING-FREE, SINGLE-RETRIEVAL decision-and-reasoning process.

The video clips are short. Therefore:

- At most ONE retrieval is allowed.
- If RETRIEVED_CONTEXT is empty:
  → You must either answer immediately OR request ONE retrieval.
- If RETRIEVED_CONTEXT already exists:
  → You MUST answer.
  → You are NOT allowed to request further retrieval.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

(1) QUESTION  
A natural-language question about an egocentric video.

IMPORTANT:
- “c” or similar references always refer to the camera wearer (“I”).
- Questions frequently involve:
  • object state before/after actions
  • causal reasoning
  • preconditions
  • counterfactual reasoning
  • first/last action
  • visibility to another person

(2) OPTIONS  
Multiple-choice options. Exactly ONE is correct.

(3) GLOBAL_CAPTIONS  
1-minute egocentric summaries describing actions, objects, temporal structure.

(4) RETRIEVED_CONTEXT (OPTIONAL)  
Additional retrieved captions. May be empty.

------------------------------------------------------------
Reasoning Requirements
------------------------------------------------------------

EgoTaskQA and QAEgo4D require:

- Precise temporal ordering
- State BEFORE and AFTER actions
- Causal links between actions and attribute changes
- Preconditions and enabling conditions
- Counterfactual feasibility reasoning
- Visibility and awareness reasoning

You must reason explicitly about:

- Which action is first/last
- Whether an attribute change is caused by a specific action
- Whether a precondition is satisfied
- Whether a counterfactual scenario is logically possible
- Whether visibility/awareness is supported by evidence

------------------------------------------------------------
Decision Rules
------------------------------------------------------------

CASE 1 — RETRIEVED_CONTEXT is EMPTY

You may ANSWER immediately ONLY IF:

- The temporal reference (first/last/before/after) is clearly resolved,
- The object state before/after is known (if required),
- The causal relation is explicitly supported,
- Counterfactual feasibility (if present) is logically grounded,
- Exactly ONE option remains consistent.

If any part of the causal chain is unclear → request retrieval.

------------------------------------------------------------

CASE 2 — RETRIEVED_CONTEXT EXISTS

You MUST answer.

Even if some ambiguity remains:
- Choose the option most consistent with available evidence.
- Do NOT invent new evidence.
- Do NOT request another retrieval.

------------------------------------------------------------
Retrieval Requirements
------------------------------------------------------------

If requesting retrieval:

- Output ONE concise English sentence.
- The query MUST:
  • Specify the temporal segment needed,
  • Specify the object whose state/attribute must be resolved,
  • Specify whether preconditions or postconditions are required,
  • Include the answer options in parentheses.

Avoid vague queries such as:
- "What happens next?"
- "More details about the video."

Be temporally and causally precise.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

Case 1 — Answer

{
  "decision": "answer",
  "answer": "<exact option text>",
  "reasoning": "<concise explanation grounded in GLOBAL_CAPTIONS and RETRIEVED_CONTEXT if any>"
}

Case 2 — Need Retrieval (only if RETRIEVED_CONTEXT is empty)

{
  "decision": "need_retrieval",
  "retrieval_query": "<one-sentence causal-temporal retrieval query including options>"
}

------------------------------------------------------------
Important Notes
------------------------------------------------------------

- Only ONE retrieval is allowed.
- If RETRIEVED_CONTEXT exists → MUST answer.
- GLOBAL_CAPTIONS are summaries, not exhaustive logs.
- Absence of evidence is NOT evidence of absence.
- All reasoning must follow explicit temporal and causal logic.
- Output ONLY the JSON object.

============================================================
In-Context Examples (EgoTaskQA + QAEgo4D Aligned)
============================================================

------------------------------------------------------------
Example 1 — Direct Answer (State Change)
------------------------------------------------------------

QUESTION:
Did the attribute of microwave change because of the action closing something?

OPTIONS:
A. yes
B. no
C. poweredness
D. edible
E. Close sink

GLOBAL_CAPTIONS:
- "I open the microwave door."
- "I place food inside and close the microwave door."
- "The microwave continues running."

RETRIEVED_CONTEXT: (empty)

OUTPUT:
{
  "decision": "answer",
  "answer": "B. no",
  "reasoning": "Closing the microwave door does not change its operational state. The captions indicate it continues running, so no attribute change occurs."
}

------------------------------------------------------------
Example 2 — Retrieval Required (Precondition Unclear)
------------------------------------------------------------

QUESTION:
Does the first action did before the person get something from something fulfill the preconditions of the action closing something?

OPTIONS:
A. no
B. yes
C. poweredness
D. edible
E. Close sink

GLOBAL_CAPTIONS:
- "I reach toward a drawer."
- "I later close something."

RETRIEVED_CONTEXT: (empty)

OUTPUT:
{
  "decision": "need_retrieval",
  "retrieval_query": "What exactly is the first action before I get something from something, and does it establish the necessary state required for closing something? (Options: A. no; B. yes; C. poweredness; D. edible; E. Close sink)"
}

------------------------------------------------------------
Example 3 — Must Answer After Retrieval
------------------------------------------------------------

QUESTION:
If the person did not open something, is the person able to put something to something?

OPTIONS:
A. no
B. yes
C. poweredness
D. edible
E. Close sink

GLOBAL_CAPTIONS:
- "I open a container."
- "I take an object out and place it onto a plate."

RETRIEVED_CONTEXT:
- "The object was inside the closed container before opening."

OUTPUT:
{
  "decision": "answer",
  "answer": "A. no",
  "reasoning": "The object was inside a closed container. Without opening it, the object could not be accessed or placed, so the action would not be possible."
}
"""

OFFLINE_PROMPTS["proactive_service_prompt_without_memory"] = """
You are an egocentric video question-answering assistant
designed for the EgoTaskQA and QAEgo4D benchmarks.

This is a TRAINING-FREE, DIRECT-ANSWER reasoning process.

The video clips are short.
You must answer the question directly
based only on the provided captions.

No retrieval is allowed.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

(1) QUESTION  
A natural-language question about an egocentric video.

IMPORTANT:
- “c” or similar references always refer to the camera wearer (“I”).
- Questions frequently involve:
  • object state before/after actions
  • causal reasoning
  • preconditions
  • counterfactual reasoning
  • first/last action
  • visibility to another person

(2) OPTIONS  
Multiple-choice options. Exactly ONE option is correct.

(3) GLOBAL_CAPTIONS  
1-minute egocentric summaries describing actions,
objects, temporal structure, and state transitions.

------------------------------------------------------------
Reasoning Requirements (STRICT)
------------------------------------------------------------

EgoTaskQA and QAEgo4D require:

- Precise temporal ordering.
- Object state BEFORE and AFTER actions.
- Explicit causal links between actions and state changes.
- Verification of preconditions and enabling conditions.
- Counterfactual feasibility reasoning.
- Visibility and awareness reasoning when another person is involved.

You MUST explicitly reason about:

- Which action occurs first or last.
- Whether an object’s state changes after a specific action.
- Whether a necessary action occurred before another action.
- Whether a counterfactual scenario is logically possible
  given the described sequence.
- Whether visibility or awareness is supported by the captions.

------------------------------------------------------------
Causal Completeness Constraint
------------------------------------------------------------

You may select an answer ONLY IF:

- The relevant temporal reference (before/after/first/last) is resolved,
- The required object states are described in the captions,
- The causal relation is supported by the captions,
- Counterfactual feasibility (if required) is logically grounded,
- All other options can be reasonably ruled out.

If the captions are incomplete or ambiguous:

- Choose the option MOST consistent with the available evidence.
- Do NOT invent new events or states.
- Do NOT assume unseen actions occurred.

------------------------------------------------------------
Forbidden Reasoning
------------------------------------------------------------

You MUST NOT:

- Infer events not mentioned in captions.
- Assume hidden object states.
- Introduce new actions.
- Use abstract logical shortcuts.
- Ignore temporal order constraints.

All reasoning must be grounded in GLOBAL_CAPTIONS.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

{
  "answer": "<exact option text>",
  "reasoning": "<concise explanation grounded strictly in GLOBAL_CAPTIONS>"
}

- The answer must match the option text exactly.
- The reasoning must be concise but explicitly grounded.
- Output ONLY the JSON object.
- No additional text.

============================================================
In-Context Examples (EgoTaskQA + QAEgo4D Aligned)
============================================================

------------------------------------------------------------
Example 1 — State Change
------------------------------------------------------------

QUESTION:
Did the attribute of microwave change because of the action closing something?

OPTIONS:
A. yes
B. no
C. poweredness
D. edible
E. Close sink

GLOBAL_CAPTIONS:
- "I open the microwave door."
- "I place food inside and close the microwave door."
- "The microwave continues running."

OUTPUT:
{
  "answer": "B. no",
  "reasoning": "The captions indicate that after I close the microwave door, it continues running. Therefore, closing the door does not change its operational state."
}

------------------------------------------------------------
Example 2 — Counterfactual
------------------------------------------------------------

QUESTION:
If the person did not open something, is the person able to put something to something?

OPTIONS:
A. no
B. yes
C. poweredness
D. edible
E. Close sink

GLOBAL_CAPTIONS:
- "I open a container."
- "I take an object out and place it onto a plate."

OUTPUT:
{
  "answer": "A. no",
  "reasoning": "The object is taken from inside a container after I open it. Without opening the container, the object would not be accessible for placing."
}

------------------------------------------------------------
Example 3 — Object State Before Action
------------------------------------------------------------

QUESTION:
What is the status of fork before the person put something to something using fork?

OPTIONS:
A. on table
B. in sink
C. inside drawer
D. on plate
E. in hand

GLOBAL_CAPTIONS:
- "A fork lies on the table."
- "I pick up the fork and use it to move food onto a plate."

OUTPUT:
{
  "answer": "A. on table",
  "reasoning": "Before I pick up the fork and use it, the caption states that it lies on the table."
}
"""


OFFLINE_PROMPTS["DEFAULT_ENTITY_TYPES"] = ["person", "location", "object", "event"]
OFFLINE_PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
OFFLINE_PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
OFFLINE_PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
OFFLINE_PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
OFFLINE_PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
OFFLINE_PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]


OFFLINE_PROMPTS["caption_reconstruction"] = """
You are an egocentric episodic caption rewriter
designed for the TaskEgoQA and QAEgo4D benchmarks.

This prompt is used in the RETRIEVAL STAGE
to selectively regenerate a more precise,
evidence-focused caption from a short egocentric video segment.

------------------------------------------------------------
Goal
------------------------------------------------------------

Given retrieval keywords, a short egocentric video segment,
and its original fine-grained caption,

decide WHETHER the segment contains ANY DIRECT,
VISUALLY GROUNDED evidence related to the keywords.

- If NO concrete keyword-relevant visual evidence is present:
  → Output exactly: ""

- If YES (at least one keyword is concretely grounded):
  → Output a rewritten caption emphasizing
    keyword-relevant objects, actions, and states,
    while preserving other clearly visible context.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

• Retrieval keywords (objects, concrete actions, spatial states).
• A short egocentric video segment (frames in temporal order).
• An ORIGINAL fine-grained caption.

------------------------------------------------------------
CRITICAL RELEVANCE CHECK (MANDATORY)
------------------------------------------------------------

Before writing anything, you MUST verify:

Does the segment contain ANY DIRECT, CONCRETE visual match
for ANY retrieval keyword?

A valid match requires:

1) A visible keyword OBJECT  
   (e.g., fork, drawer, microwave, kettle, plate).

OR

2) A visible keyword ACTION  
   (e.g., open drawer, close fridge, cut food, wash cup,
   press button, pour liquid, place object).

OR

3) A visible keyword STATE or spatial relation  
   (e.g., drawer open, cup on table, object in hand,
   appliance running or stopped, object visible to another person).

------------------------------------------------------------
STRICT EXCLUSION RULE
------------------------------------------------------------

You MUST NOT treat abstract matches as valid evidence.

The following DO NOT count as matches:

- words like “action”, “attribute”, “status”, “change”,
- logical terms (precondition, able to, fulfill),
- generic motion unrelated to keywords,
- scene similarity without the specific object/action.

If the concrete object or action named in the keywords
is NOT visibly present,
→ Output exactly: ""

------------------------------------------------------------
When Rewriting IS Allowed
------------------------------------------------------------

You MAY rewrite the caption ONLY IF:

- At least ONE retrieval keyword
  has a DIRECT visible instance in the frames.

The evidence must be object-level or action-level,
not abstract.

------------------------------------------------------------
Rewriting Rules
------------------------------------------------------------

If rewriting is allowed:

• Focus FIRST on keyword-relevant evidence:
  - explicit object names,
  - explicit actions,
  - visible state transitions,
  - object interactions,
  - spatial relations.

• Then include additional visible context
  only if clearly observable.

• Use the ORIGINAL caption only as reference.
  If it conflicts with frames, prefer frames.

You MUST:

- describe BEFORE and AFTER states if visible,
- describe visible hand interactions,
- mention whether objects are held, placed, opened, closed,
- describe visibility relative to another person if relevant.

You MUST NOT:

- copy the original caption verbatim,
- invent unseen objects or actions,
- infer intention, purpose, or correctness,
- describe hypothetical or counterfactual events.

------------------------------------------------------------
Caption Writing Requirements
------------------------------------------------------------

Write ONE concise first-person caption describing:

- what I am doing,
- which objects are involved,
- object states (open/closed, on table/in hand, running/stopped),
- continuation, transition, or completion if visible.

Prioritize concrete visual evidence
that could support state tracking or causal reasoning.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

If keyword-relevant evidence is present:

{
  "caption": "<first-person, factual rewritten caption>"
}

If NO keyword-relevant evidence is present:

""

No additional text.
No explanations.
No extra fields.

------------------------------------------------------------
Style Constraints
------------------------------------------------------------

• First-person ("I").
• Factual and observational.
• Short paragraph (1–3 sentences).
• No speculation.
• No interpretation.
• No task labels.
• No abstract reasoning terms.

------------------------------------------------------------
Important Notes
------------------------------------------------------------

• This stage functions as a STRICT FILTER + REFINER.
• Dropping weakly related segments is correct behavior.
• Only segments containing explicit, concrete evidence
  should survive.
• Precision is more important than recall.

------------------------------------------------------------
Inputs
------------------------------------------------------------
"""


OFFLINE_PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """
------------------------------------------------------------
- Goal (TaskEgoQA + QAEgo4D → Event / Entity Memory Retrieval Query) -
------------------------------------------------------------

Given a first-person (egocentric) multiple-choice question
from TaskEgoQA or QAEgo4D and its answer options,

rewrite them into EXACTLY ONE concise English declarative sentence
that can retrieve relevant EVENTS, ENTITIES, and RELATIONS
from an egocentric memory system
(e.g., event-centric knowledge graph, entity records, caption memory).

The output is NOT an answer.
It is an EVENT-ORIENTED RETRIEVAL QUERY.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I").
- Do NOT ask a question.
- Do NOT include explanations.
- Do NOT include reasoning steps.
- Do NOT include option letters.

------------------------------------------------------------
CRITICAL EVENT-RETRIEVAL PRINCIPLE
------------------------------------------------------------

This query is used for EVENT / ENTITY / RELATION retrieval,
NOT for visual similarity search.

Therefore, the query MUST:

- explicitly name concrete objects,
- explicitly describe observable actions,
- explicitly describe object states or spatial relations when relevant,
- describe event patterns that could be stored in memory,
- expand across ALL answer options.

Do NOT resolve which option is correct.
The goal is to retrieve all potentially relevant event structures.

------------------------------------------------------------
DATASET-SPECIFIC REQUIREMENT (VERY IMPORTANT)
------------------------------------------------------------

TaskEgoQA questions are highly abstract and logic-driven.

They frequently contain words such as:
- action
- attribute
- status
- change
- precondition
- able to
- fulfill
- first action
- last action
- if not

You MUST NOT include these abstract words.

Instead:

Translate them into concrete event patterns.

Examples:

- "attribute changed because of closing something"
  → I close a drawer or appliance and its door becomes closed or the device stops running.

- "if the person did not open something"
  → I open a drawer, fridge, cabinet, or container before taking or placing an object.

- "precondition of putting something"
  → I first open a container, take an object out, and then place it on a table or plate.

- "turn off something with something"
  → I press a switch or button and an appliance stops running.

- "wash something"
  → I rinse or scrub an object under a sink with water.

- "fill something using something"
  → I pour liquid into a cup or pot or hold it under a faucet.

All abstract logical structure must be grounded into visible event chains.

------------------------------------------------------------
Using Multiple-Choice Options (MANDATORY)
------------------------------------------------------------

You MUST use the answer options to EXPAND the query.

Specifically:

- Extract concrete objects mentioned in the options.
- Extract concrete actions mentioned in the options.
- Extract concrete spatial or state relations mentioned in the options.
- Merge them into ONE event-level description.
- Cover ALL visually distinct alternatives.

Do NOT:

- include abstract traits (e.g., purpose, objective),
- include logical labels (e.g., yes/no, poweredness, edible),
- include placeholder words (something, action, change, status),
- include non-visual concepts.

------------------------------------------------------------
Event-Centric Requirements
------------------------------------------------------------

The sentence SHOULD include:

- object → action → object relations,
- before/after structure expressed through concrete sequences,
- visible state transitions expressed as concrete states
  (drawer open, door closed, cup on table, appliance running or stopped),
- interactions involving another person if mentioned.

If the question refers to:
- "first action" → describe early visible actions.
- "last action" → describe later visible actions.
But do NOT use abstract wording.

------------------------------------------------------------
Temporal Wording
------------------------------------------------------------

Use phrases such as:
- at different moments,
- during early actions,
- during later actions,
- repeatedly,
ONLY when relevant.

Do NOT invent time spans.

------------------------------------------------------------
Examples (TaskEgoQA + QAEgo4D Aligned, KG-Friendly)
------------------------------------------------------------

------------------------------------------------------------
Example 1 — Attribute Change (Concrete Event Chain)
------------------------------------------------------------

Question:
"Did the attribute of microwave change because of the action closing something?
Options: A. yes; B. no; C. poweredness; D. edible; E. Close sink"

Output:
Events where I open or close a microwave door, press microwave buttons, and the microwave is visibly running or stops running after I close it.

------------------------------------------------------------

Example 2 — Counterfactual / Put and Open
------------------------------------------------------------

Question:
"If the person did not open something, is the person able to put something to something?
Options: A. no; B. yes; C. poweredness; D. edible; E. Close sink"

Output:
Events where I open a drawer, fridge, or container, take an object out, and place the object onto a table, plate, or into another container.

------------------------------------------------------------

Example 3 — Object State Before Use
------------------------------------------------------------

Question:
"What is the status of fork before the person put something to something using fork?
Options: A. on table; B. in sink; C. inside drawer; D. on plate; E. in hand"

Output:
Events where a fork is on a table, in a sink, inside a drawer, on a plate, or in my hand before I use the fork to move or place food.

------------------------------------------------------------

Example 4 — Precondition Chain
------------------------------------------------------------

Question:
"Does the action getting something from something fulfill the preconditions of the action putting something to something?
Options: A. yes; B. no; C. poweredness; D. edible; E. Close sink"

Output:
Events where I open a container, take an object from inside it, and then place that object onto a surface or into another container.

------------------------------------------------------------

Example 5 — Visibility and Interaction
------------------------------------------------------------

Question:
"Is kettle visible to the other person after the person turn off something with something?
Options: A. yes; B. no; C. poweredness; D. edible; E. Close sink"

Output:
Events where I press a switch or button to turn off an appliance and a kettle is either visible or blocked from the view of another person nearby.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------

Question: {input_text}

------------------------------------------------------------
Output:
------------------------------------------------------------
"""


OFFLINE_PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """
------------------------------------------------------------
- Goal (TaskEgoQA + QAEgo4D → Visual Embedding Retrieval Query) -
------------------------------------------------------------

Given a first-person (egocentric) multiple-choice question
from TaskEgoQA or QAEgo4D and its answer options,

rewrite them into EXACTLY ONE concise English declarative sentence
to retrieve visually relevant video segments via visual embeddings.

The output is NOT an answer.
It is a HIGH-RECALL visual retrieval query.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I").
- Do NOT output explanations.
- Do NOT output reasoning steps.
- Do NOT output multiple sentences.

------------------------------------------------------------
CRITICAL VISUAL RETRIEVAL PRINCIPLE
------------------------------------------------------------

This query is used for VISUAL EMBEDDING RETRIEVAL.

It must:

- maximize recall,
- include concrete, visually observable objects and actions,
- avoid abstract logical words,
- avoid early disambiguation,
- avoid selecting one option.

It must describe ALL possible visually relevant evidence
that could support ANY of the answer options.

------------------------------------------------------------
DATASET-SPECIFIC REQUIREMENT (VERY IMPORTANT)
------------------------------------------------------------

TaskEgoQA questions are highly abstract and logic-based.

They frequently include words such as:
- action
- first action
- last action
- attribute
- status
- precondition
- change
- able to
- fulfill
- if not

You MUST NOT include these abstract terms in the rewritten query.

Instead:

- Translate abstract templates into concrete visible actions.
- Replace “put something to something” with:
  place object on table, move object into container.
- Replace “get something from something” with:
  take object from drawer, remove object from fridge.
- Replace “turn off something with something” with:
  press button, flip switch, close appliance.
- Replace “wash something” with:
  rinse object under sink, scrub object with sponge.
- Replace “drink something with something” with:
  drink from cup, hold cup to mouth.
- Replace “fill something using something” with:
  pour liquid into container, hold container under faucet.

You MUST ground everything in concrete visual proxies.

------------------------------------------------------------
How to Use OPTIONS (MANDATORY)
------------------------------------------------------------

You MUST incorporate information from ALL options
to expand the retrieval scope.

Specifically:

- Extract concrete objects mentioned in the options.
- Extract concrete physical actions mentioned in the options.
- Extract visible spatial states mentioned in the options.
- Paraphrase them as observable visual events.
- Merge them into ONE broad but concrete retrieval query.

Do NOT:
- Copy option letters.
- List options verbatim.
- Include abstract labels (e.g., poweredness, edible).
- Include words like “attribute”, “status”, “precondition”.

------------------------------------------------------------
Visual Grounding Requirements
------------------------------------------------------------

The query MUST describe ONLY observable visual content, such as:

- My physical actions (open drawer, close fridge, cut food, wash cup).
- Interactions with concrete objects (fork, plate, microwave, kettle).
- Object states (drawer open, fridge closed, cup on table).
- Spatial relations (object on table, inside container).
- Visibility relative to another person (object visible across table).

You MUST NOT include:

- Logical terms (precondition, enable, fulfill).
- Counterfactual logic words.
- Abstract state change descriptions.
- Interpretive or conceptual words.

------------------------------------------------------------
Temporal Scope
------------------------------------------------------------

- If the question refers to “first action” or “last action”,
  describe visible moments around early or late actions,
  but do NOT use abstract temporal words.

- Do NOT invent time constraints.

------------------------------------------------------------
Examples (TaskEgoQA + QAEgo4D Aligned)
------------------------------------------------------------

------------------------------------------------------------
Example 1 — Attribute Change (Concrete Objects)
------------------------------------------------------------

Question:
"Did the attribute of microwave change because of the action closing something?
Options: A. yes; B. no; C. poweredness; D. edible; E. Close sink"

Output:
Segments where I open or close a microwave door, press microwave buttons, and the microwave is visibly running or stopped.

------------------------------------------------------------

Example 2 — Counterfactual (Put / Open)
------------------------------------------------------------

Question:
"If the person did not open something, is the person able to put something to something?
Options: A. no; B. yes; C. poweredness; D. edible; E. Close sink"

Output:
Segments where I open a drawer, fridge, or container and then take an object out and place it onto a table, plate, or into another container.

------------------------------------------------------------

Example 3 — Object State Before Action
------------------------------------------------------------

Question:
"What is the status of fork before the person put something to something using fork?
Options: A. on table; B. in sink; C. inside drawer; D. on plate; E. in hand"

Output:
Segments where a fork is on a table, in a sink, inside a drawer, on a plate, or in my hand before I use the fork to move or place food.

------------------------------------------------------------

Example 4 — Visibility to Another Person
------------------------------------------------------------

Question:
"Is kettle visible to the other person after the person turn off something with something?
Options: A. yes; B. no; C. poweredness; D. edible; E. Close sink"

Output:
Segments where I press a switch or button to turn off an appliance and a kettle is either visible or blocked from the view of another person nearby.

------------------------------------------------------------

Example 5 — Washing and Filling
------------------------------------------------------------

Question:
"Does the action filling something using something fulfill the preconditions of the action pouring from something into something?
Options: A. yes; B. no; C. poweredness; D. edible; E. Close sink"

Output:
Segments where I pour liquid into a cup, pot, or container, hold it under a faucet, and then pour the liquid into another container.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------

Question: {input_text}

------------------------------------------------------------
Output:
------------------------------------------------------------
"""


OFFLINE_PROMPTS[
    "keywords_extraction"
] = """
------------------------------------------------------------
- Goal (TaskEgoQA + QAEgo4D STRICT Visual Keyword Extraction) -
------------------------------------------------------------

Given a first-person (egocentric) multiple-choice question from the
TaskEgoQA or QAEgo4D benchmark,

extract a MINIMAL and PRECISE set of VISUAL KEYWORDS that will be used
ONLY to FILTER video segments based on DIRECT visual overlap.

These keywords are NOT for reasoning.
They are used ONLY to decide whether a video segment
contains potentially relevant visual evidence.

------------------------------------------------------------
CRITICAL VISUAL PRINCIPLE (DO NOT VIOLATE)
------------------------------------------------------------

Each extracted keyword MUST correspond to something that can be:

- directly SEEN in a single video frame, OR
- directly OBSERVED as a concrete physical action, OR
- directly OBSERVED as a concrete object state.

If a keyword cannot be visually pointed to in a frame,
it MUST NOT be included.

------------------------------------------------------------
DATASET-SPECIFIC REQUIREMENT (VERY IMPORTANT)
------------------------------------------------------------

TaskEgoQA questions are often abstract and logical.

They frequently contain words such as:
- action
- something
- attribute
- status
- first action
- last action
- precondition
- change
- objective
- able to

You MUST NOT extract these abstract words.

Instead:

- Resolve abstract references using the concrete objects and actions
  provided in the answer options.
- Only extract visually concrete objects and actions
  grounded in the options.

If the question is abstract,
the keywords MUST come primarily from the OPTIONS.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output keywords in English only.
- List keywords separated by commas.
- Do NOT output full sentences.
- Do NOT include explanations.
- Do NOT include reasoning.
- Do NOT include conclusions.
- Do NOT include option letters (A/B/C/...).
- Do NOT include abstract placeholders such as:
  something, action, activity, attribute, status, change.

------------------------------------------------------------
Mandatory Use of Multiple-Choice Options (STRICT)
------------------------------------------------------------

If the question provides answer options:

You MUST extract keywords ONLY from:

- concrete objects mentioned in the options,
- concrete physical actions mentioned in the options,
- concrete environments or visible conditions in the options.

If an option is abstract:

→ Extract ONLY its most concrete VISUAL PROXY.

Example:
- "put something to something" → extract: place object, object on table
- "turn off something with something" → extract: press button, switch off
- "washing something" → extract: wash, water, sink, sponge
- "getting something from something" → extract: take object, inside container

------------------------------------------------------------
What You MAY Extract
------------------------------------------------------------

ONLY the following categories are allowed:

1) Concrete objects  
   (fork, knife, plate, cup, microwave, fridge, kettle, cutting-board, drawer, tank, juicer, fishing-net, sink)

2) Concrete physical actions  
   (open, close, cut, wash, pour, drink, turn on, turn off, put, get, pick up, place, switch)

3) Concrete spatial or state cues  
   (on table, inside container, open door, closed drawer, visible, holding object)

------------------------------------------------------------
What You MUST NOT Extract
------------------------------------------------------------

You MUST NOT include:

- abstract task names (organizing, objective, purpose),
- logical terms (precondition, enable, fulfill),
- template words (first action, last action),
- generic placeholders (something, action, attribute),
- interpretive or evaluative words.

------------------------------------------------------------
Output Format
------------------------------------------------------------

Comma-separated keywords only.

############################################################
- Examples (TaskEgoQA + QAEgo4D Aligned) -
############################################################

------------------------------------------------------------
Example 1 — TaskEgoQA (Abstract → Option-Grounded)
------------------------------------------------------------

Question:
"If the person did not open something, is the person able to put something to something?"

Options:
A. no
B. yes
C. poweredness
D. edible
E. Close sink

Output:
open door, open drawer, open container, put object, place object, object on table

------------------------------------------------------------

Example 2 — TaskEgoQA (Attribute Change)
------------------------------------------------------------

Question:
"Did the attribute of microwave changed because of the action closing something?"

Options:
A. yes
B. no
C. poweredness
D. edible
E. Close sink

Output:
microwave, close door, microwave door, running microwave, powered on

------------------------------------------------------------

Example 3 — TaskEgoQA (Precondition Reasoning)
------------------------------------------------------------

Question:
"Does the action getting something from something fulfill the preconditions of the action putting something to something?"

Options:
A. yes
B. no
C. poweredness
D. edible
E. Close sink

Output:
get object, take object from container, inside container, put object, place object on surface

------------------------------------------------------------

Example 4 — QAEgo4D (Concrete Object State)
------------------------------------------------------------

Question:
"What is the status of fork before the person put something to something using fork?"

Options:
A. on table
B. in sink
C. inside drawer
D. on plate
E. in hand

Output:
fork, on table, in sink, inside drawer, on plate, in hand

------------------------------------------------------------

Example 5 — QAEgo4D (Visibility Reasoning)
------------------------------------------------------------

Question:
"Is kettle visible to the other person after the person turn off something with something?"

Options:
A. yes
B. no
C. poweredness
D. edible
E. Close sink

Output:
kettle, turn off switch, press button, other person, visible kettle

------------------------------------------------------------

Example 6 — TaskEgoQA (Counterfactual + Object)
------------------------------------------------------------

Question:
"If the person did not drink something with something, will cup change its status?"

Options:
A. yes
B. no
C. poweredness
D. edible
E. Close sink

Output:
drink from cup, cup in hand, cup on table, cup empty, cup filled

------------------------------------------------------------
- Real Data -
------------------------------------------------------------

Question: {input_text}

------------------------------------------------------------
Output:
------------------------------------------------------------
"""


OFFLINE_PROMPTS[
    "filtering_segment"
] = """---Role---

You are a helpful assistant to determine whether the video may contain information relevant to the knowledge based on its rough caption.
Please note that this is a rough caption of the video segments, which means it may not directly contain the answer but may indicate that the video segment is likely to contain information relevant to answering the question. 

---Video Caption---
{caption}

---Knowledge We Need---
{knowledge}

---Answer---
Please provide an answer that begins with "yes" or "no," followed by a brief step-by-step explanation.
Answer:
"""
