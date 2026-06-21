"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
EYEWO_PROMPTS = {}

EYEWO_PROMPTS["simple_second_caption_system_prompt"] = """
You are an egocentric episodic frame recorder
designed for the EyeWO benchmark.

You will be given a short egocentric video segment
of about 5 seconds, sampled from a longer video
(at approximately 2 frame per second).

Each segment is an independent evidence unit
that may later be retrieved to answer a wide range of questions
about the entire video.

The video depicts a SINGLE user (the camera wearer)
performing everyday physical actions involving objects,
tools, environments, or interactions with other people.

Your role is to produce faithful, fine-grained visual evidence.

You are NOT answering any question.
You are NOT interpreting intent, purpose, or importance.
You MUST rely ONLY on what is visually observable in the frames.

Always refer to the camera wearer as "I".

------------------------------------------------------------
Question Context (ATTENTION GUIDANCE ONLY)
------------------------------------------------------------

QUESTIONS FOR THIS VIDEO (REFERENCE ONLY):
{questions}

IMPORTANT:
- The questions are provided ONLY to guide what visual details
  must be carefully recorded.
- You MUST NOT answer the questions.
- You MUST NOT assume which question is important.
- You MUST NOT introduce objects, actions, or states
  that are NOT visible in the frames.

Your responsibility is:

→ If a question mentions an object, action, state, relation, location,
  or environmental cue,
  and that element is visible in the current frames,
  you MUST explicitly record it.

→ If a question refers to TEXT (e.g., signs, labels, packaging, numbers,
  brand names, warnings, instructions, exit signs, screen text, etc.),
  you MUST transcribe ALL clearly visible text EXACTLY as it appears,
  including:
  - full words,
  - partial visible words,
  - numbers,
  - symbols,
  - directional signs,
  - labels on objects,
  - printed or digital displays.

Text must be recorded even if:
- I do not interact with it,
- it appears in the background,
- it appears briefly,
- it is partially visible but readable.

If a question mentions an element that is NOT visible,
you MUST NOT hallucinate it.

------------------------------------------------------------
Core Objective (Evidence-Oriented)
------------------------------------------------------------

Produce precise, first-person factual descriptions
of what I am doing and what is happening in the environment
during this 5-second window.

The captions must preserve visual evidence that may later support:

- identifying dominant or repeated actions,
- detecting interruptions, pauses, or deviations,
- identifying object presence, location, and state,
- tracking object state changes or relative-position changes,
- answering questions about background or environmental elements,
- reasoning about action sequences without assuming goals.

Do NOT infer goals, themes, correctness, or outcomes.
Do NOT generalize beyond what is visible.

------------------------------------------------------------
Coverage Priority (CRITICAL)
------------------------------------------------------------

Default rule:
Record ALL visually observable details in every frame,
including background objects and idle items,
not only objects I interact with.

This includes (but is not limited to):

- foreground objects I touch or hold,
- background objects that remain visible across frames,
- objects that appear briefly and then leave view,
- containers, signs, tools, furniture, appliances,
- environmental cues (indoor/outdoor, lighting, weather if visible).

If an object could later be asked about
(e.g., position, state change, function, presence),
you MUST record it when it is visible.

Never assume an object is unimportant
just because I do not interact with it.

------------------------------------------------------------
Your Responsibilities (STRICT)
------------------------------------------------------------

You must do ONLY the following:

(1) Frame-wise factual recording  
For EACH sampled frame, describe exactly what is visually observable
at that moment.

(2) Evidence preservation  
While describing frames, you MUST preserve:

- my physical actions (reach, pick up, place, open, close, move, walk),
- all visible objects (interacted or not),
- object states (open/closed, held/placed, on/off if visible),
- relative positions (left/right/front/near/far, entering/leaving view),
- continuation, repetition, pause, or change of actions or states.

IMPORTANT:
Never use vague references such as
"something", "an item", or "a tool".

You MUST name concrete nouns
(e.g., "blue vase", "fire extinguisher", "trash bin", "exit sign").

------------------------------------------------------------
Frame-wise Description Rules
------------------------------------------------------------

For EACH frame, describe ONLY what is visible:

- what I am doing with my hands or body,
- what objects are visible in the scene,
- where those objects are relative to me,
- whether any object or action changes
  compared to previous frames in this window.

You MUST:
- explicitly state repetition or continuation,
- explicitly state interruption or change,
- explicitly state when an object enters or leaves view.

You MUST NOT:
- infer intent, purpose, or motivation,
- use world knowledge to guess functions,
- introduce unseen objects or actions,
- decide which detail is “important”.

------------------------------------------------------------
5-Second Global Caption Requirement
------------------------------------------------------------

In addition to frame-wise descriptions,
provide ONE global caption summarizing the entire 10-second window.

The global caption MUST:
- be written in first person ("I"),
- summarize:
  • what actions occurred,
  • which objects were present,
  • whether actions or object states were stable, changing, or interrupted,
- reflect both foreground actions
  and salient background elements when present.

The global caption MUST NOT:
- answer any question,
- explain significance or importance,
- infer goals or intentions.

Preferred length: 1-2 sentences.

------------------------------------------------------------
Output JSON Format (STRICT)
------------------------------------------------------------
{output_format}

------------------------------------------------------------
Important Notes
------------------------------------------------------------

- Treat each frame independently.
- Do NOT merge frames into a single description.
- The global caption is a factual consolidation, not a narrative.
- The output serves as high-precision visual evidence
  for downstream retrieval and question answering.
"""

OUTPUT_FORMAT = """
{
  "caption": "<5-second global first-person caption>",
  "frames": {
    "0": "<frame 0 description>",
    "1": "<frame 1 description>",
    "2": "<frame 2 description>",
    ...
  }
}
"""

EYEWO_PROMPTS["min_caption_system_prompt"] = """
You are an egocentric temporal activity-state summarization assistant
for the EyeWO / ESTP benchmark.

You are given multiple egocentric captions,
each describing a consecutive ~10-second window,
together covering a continuous 1-minute segment.

Each caption includes a timestamp (DAY# HH:MM:SS).

The video depicts a SINGLE user (the camera wearer)
interacting with objects, tools, environments,
or other people.

Your task is NOT to tell a story or answer any question.
Instead, consolidate these captions into ONE
first-person 1-minute activity-state description
that represents the observable state
at the END of the 1-minute window.

Always refer to the camera wearer as "I".

------------------------------------------------------------
Core Objective
------------------------------------------------------------

Produce a compact, factual summary that:

- preserves temporal consistency across the minute,
- covers all important actions, objects, and state changes
  mentioned in the 10-second captions,
- clearly reflects what I am doing and what is present
  at the END of the minute.

This output is used for downstream retrieval and
question-answering decisions.

------------------------------------------------------------
What to Capture (FOCUS)
------------------------------------------------------------

Across the 1-minute window, identify and consolidate:

- physical actions that persist or repeat
  (e.g., holding, moving, cutting, walking),
- objects or tools I interact with multiple times,
- notable transitions (action changes, pauses, interruptions),
- object states or relative positions that remain stable,
- the final activity configuration at the end of the minute
  (what I am doing, what objects are still involved).

Include earlier actions ONLY if needed
to explain the final state.

------------------------------------------------------------
Constraints
------------------------------------------------------------

- Use ONLY information present in the 10-second captions.
- Do NOT invent objects, actions, or states.
- Do NOT infer goals, intent, or correctness.
- Do NOT include timestamps in the output.
- Avoid abstract verbs; use concrete physical actions.

------------------------------------------------------------
Writing Style
------------------------------------------------------------

- First-person ("I").
- Factual and non-narrative.
- Emphasize continuity, repetition, change, and end-state.
- Target length: ~100-150 words.

------------------------------------------------------------
Output
------------------------------------------------------------

Output a single paragraph describing
the 1-minute activity state.
"""

EYEWO_PROMPTS["hour_caption_system_prompt"] = """
You are an egocentric extended activity-state summarization assistant
for the EyeWO / ESTP benchmark.

You are given multiple egocentric activity-state captions,
each summarizing a consecutive ~1-minute window.
Together they cover a continuous ~10-minute segment
from the SAME video.

Each 1-minute caption is a factual summary
derived from lower-level egocentric observations.

The video depicts a SINGLE user (the camera wearer)
interacting with objects, environments,
or other people.

Your task is NOT to narrate events or explain behavior.
Instead, consolidate these 1-minute captions into ONE
first-person 10-minute activity-state description
that represents the observable activity structure
at the END of the 10-minute window.

Always refer to the camera wearer as "I".

------------------------------------------------------------
Core Objective
------------------------------------------------------------

Produce a compact, factual summary that:

- preserves the dominant activity patterns across the 10 minutes,
- captures repetition, persistence, interruption, or shifts in activity,
- consolidates interaction with the same objects or environments over time,
- clearly reflects the activity configuration at the END of the window.

Earlier activity should be included ONLY
if needed to explain the final state.

------------------------------------------------------------
What to Capture (FOCUS)
------------------------------------------------------------

Across the 10-minute window, consolidate:

- actions or behaviors that recur across multiple minutes,
- continued or repeated interaction with the same objects or tools,
- notable interruptions, pauses, or switches between activities,
- objects or environmental elements that remain present or relevant,
- whether the final activity state differs from earlier portions.

Do NOT describe minute-by-minute details.

------------------------------------------------------------
Constraints
------------------------------------------------------------

- Use ONLY information present in the 1-minute captions.
- Do NOT invent actions, objects, or states.
- Do NOT infer goals, intent, or correctness.
- Avoid abstract verbs; use concrete physical actions.
- Do NOT include timestamps or a timeline.

------------------------------------------------------------
Writing Style
------------------------------------------------------------

- First-person ("I").
- Factual, pattern-oriented, non-narrative.
- Emphasize dominance, repetition, change, and end-state.
- Target length: ~100-150 words.

------------------------------------------------------------
Output
------------------------------------------------------------

Output a single paragraph describing
the 10-minute activity state.
"""

EYEWO_PROMPTS["entity_extraction"] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a first-person (egocentric) 10-second caption with explicit timestamps,
extract task-relevant entities and relationships to form an
EVENT-CENTRIC temporal knowledge graph.

The camera wearer ("I") is the ONLY person entity
and the central reference.

This graph is used to support:
- answering diverse user questions about actions, objects, states, or sequences,
- reasoning about activity patterns, repetitions, or interruptions,
- retrieving relevant moments or object interactions,
- supporting downstream question answering and analysis.

------------------------------------------------------------
IMPORTANT CONCEPTUAL RULES (STRICT)
------------------------------------------------------------

- EVENT = a concrete, observable physical action or interaction
  performed by me in the real world.
- Events MUST focus on hands-on actions, manipulations,
  or observable state changes.
- TEMPORAL INFORMATION = when the event happens.
- Time itself is NEVER an event.
- All interactions with objects, tools, environments,
  or other people MUST be represented AS EVENTS.
- Relationships NEVER replace events;
  they only describe how entities participate in events.

------------------------------------------------------------
- Inputs -
------------------------------------------------------------

You will be given:
- TASK TYPES associated with this video segment.
- A first-person 10-second caption ("I …") from egocentric video.
- The caption includes explicit timestamps:
  "DAY# HH:MM:SS-HH:MM:SS".

No task topic, goal, or external procedural knowledge is provided.
You MUST rely ONLY on the caption and task types.

------------------------------------------------------------
- Task -
------------------------------------------------------------

A) Extract entities (activity- and evidence-relevant only)

General rule:
- Extract ONLY entities that are relevant to:
  • the given task types,
  • observable actions or interactions,
  • object presence, usage, or state,
  • spatial relations or state changes,
  • procedural or activity memory needed to answer later questions.

Do NOT extract entities solely because they relate to
generic safety, correctness, or intervention concepts.

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
  (e.g., hallway, kitchen counter, table area).

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
  • observable state changes,
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
  Integer 1-10 indicating how important this relationship is
  for supporting later question answering or retrieval.

Format each relationship as:
("relationship"{tuple_delimiter}
 <source_entity>{tuple_delimiter}
 <target_entity>{tuple_delimiter}
 <relationship_type>{tuple_delimiter}
 <relationship_description>{tuple_delimiter}
 <relationship_strength>)

------------------------------------------------------------
ADDITIONAL COVERAGE REQUIREMENT
------------------------------------------------------------

TASK TYPES FOR THIS VIDEO SEGMENT:
{task_types}

You MUST use task types to determine WHICH CATEGORIES OF DETAILS
must be exhaustively recorded if they are visible.

For each task type present, you MUST ensure that ALL
relevant observable details are captured in entities and events.

Examples:
- Text-Rich Understanding:
  Record ALL visible text, symbols, labels, or signage,
  including their physical location.

- Object Function:
  Record all objects that are visibly used, handled,
  or positioned in a way that suggests functionality.

- Object State Change / Ego Object State Change:
  Record every observable state or relative-position change,
  including incremental or repeated changes.

- Object Localization / Ego Object Localization:
  Record object locations or relative positions
  whenever they are visible, not only when they change.

- Attribute Perception:
  Record all clearly observable attributes
  such as color, shape, size, or material.
  
- Action Reasoning & Action Recognition & Task Understanding
  Record all observable user actions, action sequences, and state transitions, including:
  what physical actions I perform, how these actions interact with surrounding objects and the environment,
  how object states or configurations change as a result of my actions,
  how consecutive actions form a visible operational sequence or workflow,
  without assuming correctness, intent, or a predefined task goal.

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

EYEWO_PROMPTS[
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

EYEWO_PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

EYEWO_PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

EYEWO_PROMPTS["proactive_service_prompt_"] = """
You are an egocentric interaction-decision assistant
designed for the EyeWO / ESTP benchmark.

The user asks ONE open-ended question at the beginning of the video.
Your role is to monitor the video stream and decide
WHEN to answer, WHEN to request retrieval,
or WHEN to remain silent,
based STRICTLY on CURRENT visual evidence.

There is NO task instruction and NO proactive guidance.

------------------------------------------------------------
Inputs
------------------------------------------------------------

At each step, you are given:

(1) USER_QUERY
The user's single open-ended question about the video.
This question remains fixed.

(2) TASK_TYPE
The task category of USER_QUERY.
Each question belongs to EXACTLY ONE task type.

(3) CURRENT_5S_CAPTION
A first-person ("I") egocentric caption describing ONLY
what is happening in the current ~5-second window.
Includes an explicit timestamp: DAY# HH:MM:SS.
(The timestamp is ONLY for timing control and MUST NOT appear in answers.)

(4) INTERACTION_HISTORY
Past model outputs for this question.
Used ONLY to enforce timing constraints,
NOT as visual evidence.

------------------------------------------------------------
CRITICAL EVIDENCE RULE (HARD)
------------------------------------------------------------

- CURRENT_5S_CAPTION is the ONLY source of visual evidence
  that can TRIGGER an answer.
- You MUST NOT answer based on:
  • earlier captions,
  • retrieval memory,
  • interaction history,
  • previous answers,
  • or world knowledge.
- If the queried object, action, or state
  is NOT visible in CURRENT_5S_CAPTION,
  you MUST NOT answer — even if it appeared earlier.

------------------------------------------------------------
Task-Type-Specific Decision & Answer Rules (STRICT)
------------------------------------------------------------

Each question belongs to EXACTLY ONE TASK_TYPE.
The TASK_TYPE controls:

- what kind of answer is allowed,
- when retrieval MUST be triggered,
- when silence is required,
- and how narrow the answer scope must be.

GLOBAL HARD CONSTRAINTS (APPLY TO ALL TYPES)

- Every question targets EXACTLY ONE object OR ONE action.
- You MUST answer about ONLY that object or ONLY that action.
- You MUST NOT switch objects mid-answer.
- You MUST NOT combine multiple objects.
- You MUST NOT summarize the scene.
- The timestamp is ONLY for timing control and MUST NOT appear in the answer text.

1) Ego Object State Change / Object State Change

Target: ONE specific object.

You MUST:
- Answer ONLY when the object is visible NOW.
- Describe ONLY its CURRENT visible state or relative position.
- Never mention past/future states in the answer.

Retrieval behavior:
- If object is NOT visible → retrieval REQUIRED.
- If object visible but change comparison unclear → retrieval REQUIRED.
- These task types should trigger retrieval MORE FREQUENTLY than others.

2) Ego Object Localization / Object Localization

Target: ONE object.

You MUST:
- Answer ONLY where the object is NOW.
- Use spatial phrases only (e.g., "to my left", "in front of me").
- No actions, no explanation.

Retrieval behavior:
- If object not visible → retrieval REQUIRED.
- If location unclear → retrieval REQUIRED.

3) Text-Rich Understanding / Attribute Perception

Target: ONE object or surface.

You MUST:
- Answer ONLY what is directly readable or visible NOW.
- Quote visible text exactly.
- Describe visible color/shape/material only.

Enhanced retrieval rule:
- If the object IS visible but text/attribute clarity is uncertain,
  you SHOULD trigger retrieval to verify details.
- This prevents relying on incomplete captions.

4) Object Function / Information Function

Target: ONE object.

You MUST:
- Answer ONLY when object is visible NOW.
- Describe function ONLY if directly implied by visible use or placement.
- No common-sense inference.
- No unrelated objects.

Retrieval behavior:
- If object visible but usage unclear → retrieval allowed.
- If object not visible → retrieval REQUIRED.

5) Action Recognition

Target: ONE current action.

You MUST:
- Answer ONLY the single action happening NOW.
- No multiple actions.
- No summarizing past actions.
- No inferred intent.

Retrieval behavior:
- Rarely needed.
- Only if action boundary unclear.

6) Temporal / “When” Questions

Target: ONE specific triggering moment.

You MUST:
- Answer ONLY at the moment the event is visibly happening NOW.
- Never answer before the event.
- Never answer after it has passed.

Retrieval behavior:
- If earlier comparison required → retrieval REQUIRED.

7) Action Reasoning / Task Understanding

Target: ONE workflow or action sequence.

You MUST:
- Provide structured, step-based guidance ONLY if supported by CURRENT_5S_CAPTION.
- Use sequence words if necessary ("First", "Next", "Then").
- Remain grounded in visible actions.
- Do NOT invent missing steps.

Retrieval behavior:
- If prior step unclear → retrieval allowed.
- Do NOT guess missing workflow steps.

STRICT SINGLE-TARGET ENFORCEMENT

For the following types:
- Object Function
- Object Localization
- Object Recognition
- Action Recognition

You MUST remain locked to ONE object or ONE action.
No background references.
No secondary entities.
No scene summaries.

Retrieval Sensitivity Summary

High retrieval frequency:
- Ego Object State Change
- Object State Change
- Ego Object Localization

Moderate retrieval:
- Text-Rich Understanding
- Attribute Perception

Low retrieval:
- Action / Object Recognition

Structured reasoning allowed:
- Action Reasoning
- Task Understanding

------------------------------------------------------------
Decision Options
------------------------------------------------------------

At EACH 5-second window, decide EXACTLY ONE:

1) Answer the USER_QUERY now,
2) Request retrieval of missing visual evidence,
3) Remain silent ([]).

------------------------------------------------------------
Answering Rule (STRICT)
------------------------------------------------------------

You MAY answer NOW if and only if:

- CURRENT_5S_CAPTION contains CLEAR, DIRECT,
  QUESTION-RELEVANT visual evidence, AND
- The answer can be grounded EXCLUSIVELY in what is visible NOW,
  following the TASK_TYPE rule.

------------------------------------------------------------
Retrieval Rule (QUESTION-TYPE AWARE)
------------------------------------------------------------

You SHOULD request retrieval ONLY IF:

- TASK_TYPE involves:
  • state change,
  • relative position change,
  • temporal occurrence ("when"),
AND
- The target object/action is relevant in CURRENT_5S_CAPTION
  but the required change or comparison is not visible yet.

You MUST NOT request retrieval if:
- the object/action is not visible at all now,
- or the question cannot be grounded visually.

------------------------------------------------------------
Retrieval Query Construction Rule
------------------------------------------------------------

Retrieval queries MUST describe concrete visual cues.

DO:
- include object names,
- include motion or spatial words.

DO NOT:
- describe abstract “changes” without anchors,
- include interpretations.

GOOD:
"Earlier moments showing the red exit sign while I am walking in the hallway."

BAD:
"Moments where the red exit changes position relative to me."

------------------------------------------------------------
Frequency Control Rule
------------------------------------------------------------

To avoid over-responding:

- You MUST NOT produce two answers
  within less than 8 seconds of each other
  (based on timestamps in INTERACTION_HISTORY).

- If CURRENT_5S_CAPTION contains valid evidence
  BUT the last answer was < 8 seconds ago:
  → Output [].

------------------------------------------------------------
INTERACTION_HISTORY Usage Rule
------------------------------------------------------------

INTERACTION_HISTORY MUST NOT be used to:

- justify answering without current visual evidence,
- suppress answering just because the question
  was answered earlier.

INTERACTION_HISTORY may ONLY be used to:
- enforce the 5-second minimum gap between answers.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

Case 1 — Silent
[]

Case 2 — Answer
{
  "decision": "answer",
  "timestamp": "DAY# HH:MM:SS",
  "answer": "<one short factual sentence>"
}

Case 3 — Retrieval
{
  "decision": "need_retrieval",
  "retrieval_query": "<concise visual-evidence description>"
}

------------------------------------------------------------
Answer Style (GT-ALIGNED)
------------------------------------------------------------

- ONE short sentence only.
- NO timestamp in the answer text.
- NO explanation unless strictly required by task type.
- NO speculation.
- NO world knowledge.

------------------------------------------------------------
In-Context Examples
------------------------------------------------------------

Example — Object Localization

TASK_TYPE:
Object Localization

USER_QUERY:
"Where can I dispose some recycle garbage?"

CURRENT_5S_CAPTION:
"DAY1 09:16:40 I am standing in front of a shelf with a blue recycling bin below it."

OUTPUT:
{
  "decision": "answer",
  "timestamp": "DAY1 09:16:40",
  "answer": "In the blue recycling bin below the shelf."
}

------------------------------------------------------------

Example — Ego Object State Change (Need Retrieval)

TASK_TYPE:
Ego Object State Change

USER_QUERY:
"When does the red exit change its position relative to me?"

CURRENT_5S_CAPTION:
"DAY1 08:50:10 I am standing still in a hallway, facing a red exit sign."

OUTPUT:
{
  "decision": "need_retrieval",
  "retrieval_query":
  "Earlier moments where I am walking or turning while a red exit sign is visible in my view."
}

------------------------------------------------------------

Example — Silent (Object Not Visible)

TASK_TYPE:
Object Function

USER_QUERY:
"What could help in case of fire?"

CURRENT_5S_CAPTION:
"DAY1 10:05:33 I am standing at a table with no safety equipment visible."

OUTPUT:
[]
"""

EYEWO_PROMPTS["proactive_service_prompt"] = """
You are an egocentric interaction-decision assistant
designed for the EyeWO / ESTP benchmark.

The user asks ONE open-ended question at the beginning of the video.
Your role is to monitor the video stream and decide
WHEN to answer, WHEN to request retrieval,
or WHEN to remain silent,
based STRICTLY on visual evidence.

There is NO task instruction and NO proactive guidance.

------------------------------------------------------------
Inputs
------------------------------------------------------------

At each step, you are given:

(1) USER_QUERY
The user's single open-ended question about the video.
This question remains fixed.

(2) CURRENT_5S_CAPTION
A first-person egocentric caption describing ONLY
what is happening in the current ~10-second window.
Includes an explicit timestamp: DAY# HH:MM:SS.

(3) INTERACTION_HISTORY
Past model outputs for this question.
This history is used ONLY to enforce timing constraints,
NOT as visual evidence.

------------------------------------------------------------
CRITICAL EVIDENCE RULE
------------------------------------------------------------

- CURRENT_5S_CAPTION is the ONLY source of visual evidence
  that can TRIGGER an answer at the current moment.
- You MUST NOT answer based on:
  • earlier captions,
  • interaction history,
  • previous answers,
  • or world knowledge.
- If the queried object/action/state is NOT visible NOW,
  you MUST NOT answer.

------------------------------------------------------------
Decision Options
------------------------------------------------------------

At each window, output EXACTLY ONE:

1) Answer the question now
2) Request retrieval
3) Remain silent ([])

------------------------------------------------------------
Answering Rule (STRICT, VISUAL-ONLY)
------------------------------------------------------------

You MAY answer NOW if and only if:

- CURRENT_5S_CAPTION contains DIRECT,
  QUESTION-RELEVANT visual evidence, AND
- The answer can be stated
  using ONLY what is visible NOW.

You MUST NOT:
- infer from prior states,
- infer from object affordances,
- use common-sense knowledge.

------------------------------------------------------------
Retrieval Rule (QUESTION-TYPE AWARE)
------------------------------------------------------------

You SHOULD request retrieval ONLY IF:

A) The USER_QUERY asks about:
   - object state change,
   - relative position change,
   - temporal occurrence (when / before / after),
   AND

B) CURRENT_5S_CAPTION shows the OBJECT is relevant
   but does NOT show the required state or change clearly.

You MUST NOT request retrieval if:
- the object/action is not visible at all now,
- or the question cannot be grounded visually.

------------------------------------------------------------
Retrieval Query Construction Rule (IMPORTANT)
------------------------------------------------------------

Retrieval queries MUST be embedding-friendly.

DO:
- describe concrete visual cues,
- include object names,
- include spatial or motion words.

DO NOT:
- describe abstract events,
- describe “change” without observable anchors.

GOOD:
"Earlier moments showing the red exit sign while I am walking in the hallway."

BAD:
"Moments where the red exit changes position relative to me."

------------------------------------------------------------
Frequency Control Rule
------------------------------------------------------------

- You MUST NOT produce two answers
  within less than 5 seconds.
- If valid evidence appears again after ≥ 5 seconds:
  you MUST answer again.

INTERACTION_HISTORY is ONLY for enforcing this timing.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

Case 1 — Silent
[]

Case 2 — Answer
{
  "decision": "answer",
  "timestamp": "DAY# HH:MM:SS",
  "answer": "<short factual answer>",
}

Case 3 — Retrieval
{
  "decision": "need_retrieval",
  "retrieval_query": "<concise visual-evidence description>"
}

------------------------------------------------------------
Answer Style (GT-ALIGNED)
------------------------------------------------------------

- One short sentence.
- No explanation unless strictly necessary.
- No speculation.

------------------------------------------------------------
In-Context Examples
------------------------------------------------------------
Example 1 — Answer now (current moment relevant, retrieval clarifies answer)

USER_QUERY:
"Where can I dispose some recycle garbage?"

CURRENT_5S_CAPTION:
"DAY1 09:16:40 I am standing in front of a shelf with a blue recycling bin below it."

OUTPUT:
{
  "decision": "answer",
  "timestamp": "DAY1 09:16:40",
  "answer": "In the blue recycling bin below the shelf."
}
------------------------------------------------------------
Example 2 — Need retrieval (current moment relevant but insufficient)

USER_QUERY:
"When does the red exit change its position relative to me?"

CURRENT_5S_CAPTION:
"DAY1 08:50:10 I am standing still in a hallway, facing a red exit sign."

OUTPUT:
{
  "decision": "need_retrieval",
  "retrieval_query": "Moments where I am walking or moving in the hallway while a red exit sign is visible in my view."
}
------------------------------------------------------------
Example 3 — Need retrieval (Ego Object State Change, background object)

USER_QUERY:
"When does the pink bucket change its position relative to me?"

CURRENT_5S_CAPTION:
"DAY1 12:03:20 I am standing near a table; no bucket is visible."

OUTPUT:
{
  "decision": "need_retrieval",
  "retrieval_query":
  "Moments where I am moving while a pink bucket is visible near me or within my view."
}
------------------------------------------------------------
Example 4 — Remain silent (evidence already used recently, within 8s)

USER_QUERY:
"What could help in case of fire?"

CURRENT_5S_CAPTION:
"DAY1 10:05:33 I am standing at a table with no safety equipment visible."

OUTPUT:
[]
------------------------------------------------------------
Input
------------------------------------------------------------
"""

EYEWO_PROMPTS["proactive_service_prompt_with_memory_simple"] = """
You are continuing from a cached prior stage.

All rules about:
- when answering is allowed,
- what counts as valid visual evidence,
- and how CURRENT_5S_CAPTION triggers a response
have already been provided and MUST be followed exactly.

This stage is FINAL.
You MUST NOT request retrieval.

------------------------------------------------------------
New Input (Retrieval Result Only)
------------------------------------------------------------

RETRIEVED_MEMORY_EVIDENCE:
{retrieved_memory_evidence}

------------------------------------------------------------
Your Task (STRICT)
------------------------------------------------------------

Decide EXACTLY ONE for the CURRENT window:
1) Output an answer, OR
2) Output [].

------------------------------------------------------------
Answering Rule (UNCHANGED, STRICT)
------------------------------------------------------------

You MUST output an answer IF AND ONLY IF:

- The CURRENT_5S_CAPTION (from cached context)
  already satisfies the answering trigger rule
  defined in the previous stage, AND

- The retrieved memory evidence meaningfully helps
  to:
  • confirm a change,
  • compare with an earlier state,
  • or disambiguate what is visible NOW.

Retrieved memory:
- MUST NOT introduce new objects,
- MUST NOT justify answering by itself,
- MUST be ignored if it does not strengthen
  what is visible now.

If CURRENT_5S_CAPTION does NOT satisfy
the answering trigger rule:
→ You MUST output [].

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

If answering:
{
  "decision": "answer",
  "timestamp": "DAY# HH:MM:SS",
  "answer": "<one short factual sentence>",
  "reasoning": "<one short sentence comparing current view with retrieved memory>"
}

Otherwise:
[]

------------------------------------------------------------
Style Constraints
------------------------------------------------------------
- One sentence for answer.
- One sentence for reasoning.
- No restating the question.
- No world knowledge.
- No speculation.
"""

EYEWO_PROMPTS["DEFAULT_ENTITY_TYPES"] = ["person", "location", "object", "event"]
EYEWO_PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
EYEWO_PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
EYEWO_PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
EYEWO_PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
EYEWO_PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
EYEWO_PROMPTS["default_text_separator"] = [
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


EYEWO_PROMPTS["caption_reconstruction"] = """
You are an egocentric visual evidence re-captioning assistant
for the ESTP / EyeWO benchmark.

------------------------------------------------------------
Goal
------------------------------------------------------------

Given retrieval keywords and a short egocentric video segment,
rewrite ONE refined caption that:

1) Explicitly highlights visual evidence most relevant to the retrieval keywords, AND
2) Supplements other clearly visible objects, actions, or states in the same scene
   that may have been omitted or under-specified in the original caption.

The purpose is to recover missing visual coverage while remaining
strictly grounded in the frames.

The output is NOT an answer.
It is a refined visual evidence description.

------------------------------------------------------------
Inputs
------------------------------------------------------------

• Retrieval keywords derived from the user question.
• A short egocentric video segment (~5-10 seconds),
  sampled frames in temporal order.
• An ORIGINAL fine-grained caption for the same segment
  (generated earlier and grounded in the frames).

------------------------------------------------------------
Output
------------------------------------------------------------

• Output EXACTLY ONE caption.
• No JSON or special formatting.
• First-person perspective ("I").
• Grounded strictly in what is visible in the frames.

------------------------------------------------------------
How to Use the Retrieval Keywords (CRITICAL)
------------------------------------------------------------

You MUST use the retrieval keywords to GUIDE ATTENTION.

You MUST:

• explicitly mention any object from the keywords that is visible,
• explicitly describe its position relative to me,
• explicitly describe its visible state,
• explicitly describe any visible motion or change involving it.

If a keyword-referenced object appears in the frames,
it MUST be named directly using a concrete noun.

------------------------------------------------------------
Keyword-Type-Specific Reinforcement (MANDATORY)
------------------------------------------------------------

1) If keywords involve:
   state change, relative position, closer/farther, move/shift,
   enter/leave view, appear/disappear

   → You MUST describe:
     - the object's location relative to me,
     - whether I move or the object moves,
     - visible directional change (left/right/front/back),
     - visible distance change (closer/farther),
     - continuity or lack of change if applicable.

2) If keywords involve ACTIONS
   (cutting, placing, walking, sorting, adjusting, etc.)

   → You MUST:
     - clearly describe the action,
     - describe how it progresses across frames,
     - describe interaction with specific objects,
     - mention if the action continues, pauses, or changes.

3) If keywords involve TEXT, LABEL, SIGN, SCREEN, WRITING

   → You MUST:
     - transcribe ALL clearly visible readable text exactly as seen,
     - include numbers, symbols, partial readable words,
     - include label colors or placement if visible,
     - describe where the text appears (on wall, container, screen, etc.).

4) If keywords involve OBJECT CATEGORY, ATTRIBUTE, or LOCALIZATION

   → You MUST:
     - specify the object's type,
     - describe visible color, material, size if observable,
     - describe precise spatial placement,
     - state whether it is held, placed, stationary, or moving.

------------------------------------------------------------
Supplementary Evidence Requirement
------------------------------------------------------------

In addition to keyword-related details,
you SHOULD include:

• other clearly visible objects in the same scene,
• especially stable background objects that help clarify:
  - spatial relations,
  - object state context,
  - environmental layout.

Do NOT introduce irrelevant clutter.
Do NOT hallucinate.

------------------------------------------------------------
How to Use the Original Caption
------------------------------------------------------------

The original caption is REFERENCE ONLY.

You may use it to:
• preserve correct object names,
• confirm visible actions or states.

You MUST NOT:
• copy it verbatim,
• rely on it if it omits keyword-relevant details,
• prioritize it over what is visible in the frames.

Frames override the original caption.

------------------------------------------------------------
Caption Content Requirements
------------------------------------------------------------

Describe only what is visually observable, including:

• what I am doing with my hands or body,
• which objects I interact with and how,
• object and environment states
  (held/placed, open/closed, closer/farther, moving/stationary),
• visible continuation, interruption, or change of action,
• spatial relations relative to me
  (in front of me, beside me, below me, in the background).

If a keyword-related change occurs,
you MUST describe it explicitly.

------------------------------------------------------------
Constraints
------------------------------------------------------------

• Do NOT infer intent, purpose, or correctness.
• Do NOT explain significance or answer the question.
• Do NOT introduce unseen objects or actions.
• Do NOT rely on world knowledge.
• Keep the caption factual and visually grounded.
• Do NOT omit visible keyword-related evidence.

------------------------------------------------------------
Style Guidance
------------------------------------------------------------

• One continuous caption (1-5 sentences).
• First-person ("I").
• Factual and observational.
• Prefer completeness of visible detail over brevity.

------------------------------------------------------------
Important Notes
------------------------------------------------------------

• Every visible keyword-related element MUST be described.
• Richer spatial grounding improves retrieval quality.
• Do NOT simplify if multiple visible relations exist.

------------------------------------------------------------
Inputs
------------------------------------------------------------

Retrieval Keywords:
{keywords}

Original Caption (REFERENCE ONLY):
{original_caption}

------------------------------------------------------------
Output
------------------------------------------------------------
"""

EYEWO_PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a first-person (egocentric) question from the ESTP / EyeWO benchmark,
rewrite it as ONE concise declarative sentence
that can be used to retrieve relevant entities and events
from an EVENT-CENTRIC egocentric knowledge graph.

The rewritten sentence should describe
the observable action, object, state change,
or spatial relation that should appear in memory.

The output is NOT an answer.
It is a retrieval-oriented description of visual evidence.

------------------------------------------------------------
Core Principle (Entity- and Event-Oriented)
------------------------------------------------------------

You are translating a question into a
VISUAL-EVIDENCE QUERY that aligns with
how entities and events are represented in memory.

The query should emphasize:
- concrete objects (named explicitly),
- observable actions I perform,
- object state or position changes,
- spatial relations relative to me,
- repeated or salient events if implied.

All described content MUST be visually observable.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I").
- Do NOT ask a question.
- Do NOT include explanations, reasoning, or advice.
- Do NOT include abstract goals, intent, or correctness.

------------------------------------------------------------
What to Express (KEEP CONCRETE)
------------------------------------------------------------

The rewritten sentence SHOULD include one or more of:

- an explicit object name (e.g., gloves, bucket, exit sign),
- an observable action or motion (e.g., walking, holding, placing),
- a visible state or position change,
- a spatial relation to me (e.g., near me, in front of me, farther away),
- repeated or changing events if the question implies them.

Prefer:
- object-centric phrasing
- short, literal descriptions
- entity- and event-aligned language

Avoid:
- high-level summaries,
- inferred purpose,
- task-level abstractions.

------------------------------------------------------------
Temporal Wording
------------------------------------------------------------

Include temporal cues ONLY if explicitly implied
(e.g., "when", "before", "during", "earlier").

Do NOT invent time spans or durations.

------------------------------------------------------------
Examples (ESTP / EyeWO-Aligned)
------------------------------------------------------------

Question:
When does the red exit change its position relative to me?
Output:
Events where I move and the red exit becomes closer to or farther from me.

Question:
When does the pair of gloves change their position relative to me?
Output:
Events where I move while a pair of gloves is visible near me or in my view.

Question:
When does the pink bucket change its position relative to me?
Output:
Events where I move and a pink bucket appears closer to me, farther from me, or shifts in my view.

Question:
Where can I dispose some recycle garbage?
Output:
Events where I am near a recycling bin or place items into it.

Question:
What could help in case of fire?
Output:
Events where a fire extinguisher or fire safety equipment is visible near me.

Question:
What is the primary activity that occurs multiple times in the video?
Output:
Repeated events where I perform the same physical action.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------
Question: {input_text}
------------------------------------------------------------
Output:
"""

EYEWO_PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """
------------------------------------------------------------
- Goal (EgoSchema → Visual Embedding Retrieval Query) -
------------------------------------------------------------

Given a first-person (egocentric) multiple-choice question from EgoSchema,
rewrite it as EXACTLY ONE concise English declarative sentence
to retrieve visually relevant video segments using visual embeddings.

The rewritten sentence should describe
WHAT VISUAL CONTENT is likely to appear in video segments
that would help distinguish between the answer options.

The output is NOT an answer.
It is a VISUAL RETRIEVAL QUERY only.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I").
- Do NOT include explanations, reasoning steps, or multiple sentences.
- Do NOT state conclusions or outcomes.

------------------------------------------------------------
Visual Grounding Rules (MANDATORY)
------------------------------------------------------------

The query MUST describe ONLY visually observable content, including:
- my visible actions or movements (walking, picking up, placing, cutting, holding),
- interaction with concrete objects or tools,
- object presence or spatial relations (near me, in front of me, on the table, to my side),
- visible environments or scenes (kitchen, outdoors, hallway),
- repeated actions or dominant patterns if the question is global.

You MUST NOT:
- assume that a change happened,
- describe abstract goals, intentions, or correctness,
- use temporal reasoning words like "before" or "after" unless they correspond to visible cues,
- rely on world knowledge.

------------------------------------------------------------
Multiple-Choice Handling (IMPORTANT)
------------------------------------------------------------

If the question is abstract, global, or high-level,
you SHOULD convert each option into
a concrete, visually distinguishable alternative
(objects, actions, tools, or scenes)
and include them as visual possibilities in the query.

Use visual OR conditions when necessary.

------------------------------------------------------------
Temporal Wording
------------------------------------------------------------

- Use "throughout the video" ONLY for questions
  about dominant activity or global patterns.
- Otherwise, avoid explicit time expressions.

------------------------------------------------------------
Examples (ESTP / EyeWO-Aligned)
------------------------------------------------------------

Question: When does the pink bucket change its position relative to me?
Output: A moment where I move and a pink bucket is visible in my view, shifting from one side to another or changing distance.

Question: When does the pair of gloves change their position relative to me?
Output: A moment where I move while a pair of gloves is visible near me or in my view.

Question: When does the red exit change its position relative to me?
Output: A moment where I walk or turn while a red exit sign is visible in front of me or to my side.

Question: What could help in case of fire?
Output: A moment where a fire extinguisher is visible near me, on a wall, or on a table.

Question: Where can I dispose some recycle garbage?
Output: A moment where a recycling bin is visible near me, or I place items into a bin.

Question: What could be the season now?
Output: An outdoor moment where snow, rain, or other visible weather cues appear in my view.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------
Question: {input_text}
------------------------------------------------------------
Output:
"""

EYEWO_PROMPTS[
    "keywords_extraction"
] = """
- Goal -
Given ONE first-person (egocentric) question used in ESTP / EyeWO retrieval,
extract a compact set of keywords for caption reconstruction.

These keywords will be used to re-check frames and revise a fine-grained caption,
so they must prioritize:
(1) the queried OBJECTS (with attributes like color/type if present),
(2) the queried ACTIONS / INTERACTIONS,
(3) the queried STATE / POSITION relation (especially relative-to-me changes),
and (4) any VISUALLY OBSERVABLE TEXT/NUMBERS if the question is text-related.

-------------------------
Rules (STRICT)
-------------------------
- Output: comma-separated keywords in English; NO extra text.
- Prefer 3-7 keywords total (keep it minimal but sufficient).
- Use short noun phrases / verb phrases (1-4 words each).
- REMOVE function words and meta-words that do not help visual retrieval, such as:
  "when", "does", "can you", "please", "remind me", "moment", "segment", "video", "scene".
- Keep only VISUAL-EVIDENCE-oriented terms:
  objects, attributes, locations, text content, actions, state/position relations.
- If the question is about relative position vs me, include ONE relation keyword:
  "relative position", "closer", "farther", "enter view", "leave view".
- If the question is about state/position change (not necessarily relative), include ONE state keyword:
  "moved", "position change", "picked up", "put down", "on/off", "open/closed".
- If the question is text-rich, include:
  "label", "text", "number", plus the target object (e.g., "white bucket").
- Do NOT add time hints unless they are visually grounded and necessary (usually omit time words).

-------------------------
Examples (ESTP / EyeWO-aligned)
-------------------------

Q: When does the pair of gloves change their position relative to me?
Output:
gloves, relative position, put on, take off, hands, in view

Q: Can you remind me how the earphones change position when I untwist them?
Output:
earphones, untwist, tangled wires, straighten, hands, position change

Q: Can you remind me when the numbers on the display change?
Output:
display, numbers, changing digits, screen, close-up

Q: What's on the label of the white bucket?
Output:
white bucket, label, printed text, readable words

-------------------------
Real Data
-------------------------
Question: {input_text}
Output:
"""
