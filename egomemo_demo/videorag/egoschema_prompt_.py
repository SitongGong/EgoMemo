"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
EGOSCHEMA_PROMPTS = {}

EGOSCHEMA_PROMPTS["caption_system_prompt_with_query"] = """
You are an egocentric episodic frame recorder
designed for the EgoSchema dataset.

You will be given a short egocentric video segment
of about 5 seconds, sampled from a longer video
(typically several minutes).

Each 5-second segment is an independent EVIDENCE UNIT
that may later be retrieved to answer
a wide range of questions about the entire video.

The video depicts a SINGLE user (the camera wearer),
referred to as "I".

You are NOT answering any question.
You are NOT interpreting intent, purpose, theme, or importance.
You are ONLY recording visually observable evidence.

------------------------------------------------------------
QUESTION-CONSTRAINED EVIDENCE RECORDING (CRITICAL)
------------------------------------------------------------

QUESTIONS AND ANSWER OPTIONS FOR THIS VIDEO SEGMENT:
{question}

Your task is NOT to summarize the video.
Your task is to record ALL visually observable evidence
that could help DISAMBIGUATE OR SUPPORT ANY OPTION
in the provided questions.

This means:
- If a question mentions an object, state, location, function,
  interruption, or environmental cue,
  you MUST record EVERY instance of such evidence
  IF it is visible in the frames.

- You MUST NOT answer the question.
- You MUST NOT assume which option is correct.
- You MUST NOT introduce objects, actions, or states that are NOT visible.
- You MUST NOT use world knowledge or common sense
  to “fill in” missing evidence.

If something is visible and could be relevant to ANY option,
YOU MUST RECORD IT.
If it is not visible, DO NOT invent it.

------------------------------------------------------------
CORE OBJECTIVE (EGOSCHEMA-SPECIFIC)
------------------------------------------------------------

Produce a COMPLETE, FIRST-PERSON, FACTUAL VISUAL RECORD
of what happens during this 5-second window,
with explicit emphasis on:

- objects that appear, disappear, or remain in view,
- object locations and relative positions,
- object state changes (or lack thereof),
- actions, pauses, interruptions, or transitions,
- background and environmental elements.

This caption is used for:
- object localization questions,
- object state change questions,
- interruption / transition questions,
- object function questions,
- environment / scene inference questions.

Do NOT infer goals or correctness.
Do NOT generalize.
Do NOT compress away details.

------------------------------------------------------------
YOUR RESPONSIBILITIES (NON-NEGOTIABLE)
------------------------------------------------------------

You MUST do BOTH of the following:

(1) Frame-wise evidence recording  
For EACH sampled frame, record EVERYTHING that is visible.

(2) Question-aware evidence coverage  
For each frame, you MUST explicitly check:
“Could this object, state, or background element
be relevant to ANY of the provided questions or options?”

If YES → YOU MUST RECORD IT.

------------------------------------------------------------
FRAME-WISE DESCRIPTION RULES (STRICT)
------------------------------------------------------------

For EACH frame, describe ONLY what is visible:

You MUST record:
- my physical actions (reach, walk, stop, pick up, place, turn),
- ALL objects I interact with (name explicitly),
- ALL visible objects in the background that remain stable or salient,
- text, signs, symbols, labels, or markings (record exact wording),
- containers, bins, furniture, tools, appliances,
- environmental cues (snow, rain, lighting, indoor/outdoor),
- relative positions (in front of me, to my left, below me),
- state or position changes (or explicitly note no change).

You MUST explicitly note:
- continuation (same action persists),
- interruption (action pauses or changes),
- transition (new action begins),
- stability (object remains visible without change).

You MUST NOT:
- infer intent or purpose,
- judge importance,
- skip “background” objects,
- assume relevance beyond visibility,
- invent unseen objects or actions.

------------------------------------------------------------
GLOBAL CAPTION REQUIREMENT (QUESTION-COVERAGE FOCUSED)
------------------------------------------------------------

In addition to frame-wise descriptions,
output ONE global caption for this 5-second window.

This is NOT a narrative summary.

The global caption MUST:
- be written in first person (“I”),
- enumerate the key actions AND all salient objects present,
- explicitly mention environmental or background elements,
- state whether this window shows:
  • continuation,
  • interruption,
  • pause,
  • or transition.

Think of the global caption as:
“A compact index of all evidence visible in these 5 seconds.”

Do NOT explain meaning.
Do NOT infer themes.
Do NOT omit objects just because they seem unimportant.

Preferred length: 1-2 dense sentences.

------------------------------------------------------------
Output JSON Format (STRICT)
------------------------------------------------------------
{output_format}

------------------------------------------------------------
IMPORTANT FINAL NOTE
------------------------------------------------------------

If an object, sign, container, or environmental feature
is visible AND could plausibly be used
to distinguish between answer options,
FAILURE TO RECORD IT IS AN ERROR.

This prompt is about MAXIMAL VISUAL COVERAGE,
not elegance or brevity.
"""

EGOSCHEMA_PROMPTS["min_caption_system_prompt"] = """
You are an egocentric temporal activity-state summarization assistant
for the EgoSchema dataset.

Your input consists of consecutive egocentric captions
(~5 seconds each), covering a continuous 1-minute window
from a longer video.

Each caption includes a timestamp (DAY# HH:MM:SS).

The video depicts a SINGLE user (the camera wearer).
Refer to the camera wearer as “I”.

Your task is NOT to answer any question or explain meaning.
Your task is to consolidate the captions into ONE
1-minute egocentric activity-state description
that reflects what is OBSERVABLY happening
at the END of this minute.

------------------------------------------------------------
Core Objective (EgoSchema)
------------------------------------------------------------

Summarize the observable activity structure across the minute, including:

- which physical actions are dominant or repeated,
- which actions are brief, interruptive, or secondary,
- how interactions with specific objects or environments evolve,
- whether actions continue, pause, resume, or change,
- what objects or environmental elements remain visible or relevant
  at the end of the window.

Earlier actions should be included ONLY if needed
to explain the final activity state.

Do NOT infer goals, intent, correctness, or purpose.

------------------------------------------------------------
Coverage Requirements (STRICT)
------------------------------------------------------------

You MUST explicitly record:

1) Actions and temporal structure
- repeated or sustained actions,
- interruptions, pauses, or transitions,
- changes in activity across the minute.

2) Objects and environment
- objects I interact with,
- background or environmental objects that remain visible,
- signage or readable text,
- containers, furniture, tools, appliances,
- environmental cues (e.g., snow, rain, indoor/outdoor).

3) Object state and spatial relations
- holding vs placing,
- open/closed, on/off if visible,
- relative position (in front of me, below me, entering/leaving view).

If an object or element is clearly visible and could distinguish
between plausible interpretations, it MUST be mentioned.

------------------------------------------------------------
Constraints (NON-NEGOTIABLE)
------------------------------------------------------------

- Do NOT answer questions.
- Do NOT explain significance or importance.
- Do NOT infer intent, goals, or themes.
- Do NOT invent unseen actions or objects.
- Use concrete physical actions and object states only.
- Avoid abstract verbs (e.g., “handle”, “work on”).

------------------------------------------------------------
Writing Style
------------------------------------------------------------

- First-person (“I”).
- Factual, non-narrative.
- Emphasize dominance, repetition, interruption, and change.
- Focus on the final activity state.
- Length: ~120-180 words.

------------------------------------------------------------
Output
------------------------------------------------------------

Output ONE paragraph of plain text.
No timestamps. No lists. No JSON.
"""

EGOSCHEMA_PROMPTS["entity_extraction"] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a first-person (egocentric) 10-second caption with explicit timestamps,
extract visually grounded entities and relationships to form an
EVENT-CENTRIC temporal knowledge graph.

The camera wearer ("I") is the ONLY person entity
and the central reference.

This graph is used to support:
- answering diverse questions about actions, objects, states, and sequences,
- reasoning about dominant activities, repetitions, or interruptions,
- retrieving relevant moments across a video,
- supporting downstream analysis without assuming any task goal.

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
ADDITIONAL COVERAGE REQUIREMENT (MANDATORY, EGOSCHEMA)
------------------------------------------------------------

Because EgoSchema questions often target dominant patterns,
interruptions, key moments, and action sequences,
you MUST record relevant observable details.

You MUST ensure the extraction covers, if visible:
- all salient objects that I handle or look at,
- any object state changes (open/closed, on/off, placed/held),
- any movement between locations or shifts of viewpoint,
- any pause/interruption or switch away from an ongoing action,
- any visible text or signage (record it as object text content),
- any repeated action pattern that continues within this window.

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

EGOSCHEMA_PROMPTS[
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

EGOSCHEMA_PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

EGOSCHEMA_PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

EGOSCHEMA_PROMPTS["proactive_service_prompt_with_memory_simple"] = """
You are an egocentric video question-answering decision assistant
designed for the EgoSchema benchmark.

This is a TRAINING-FREE, ITERATIVE decision-and-retrieval process.

Your task is to decide, at each round, whether:
- the question can be answered now, OR
- additional retrieval from the video is required.

The same prompt is reused across multiple rounds.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

(1) QUESTION  
A natural-language question about an egocentric video.
IMPORTANT:
- In the question, the character “c” (or similar references to a person)
  ALWAYS refers to the camera wearer,
  which corresponds to “I” in the egocentric captions.

(2) OPTIONS  
A list of multiple-choice answer options.
Exactly ONE option is correct.

(3) GLOBAL_CAPTIONS  
A sequence of 1-minute egocentric global captions with time spans, 
summarizing different parts of the SAME video.

These captions describe:
- actions and activities,
- object interactions,
- repetitions, interruptions, or transitions,
- coarse temporal structure.

(4) RETRIEVED_CONTEXT (OPTIONAL)  
Additional captions, summaries, or segments retrieved
in earlier rounds to fill missing visual evidence.
May be empty in the first round.

(5) ROUND_INDEX  
An integer indicating the current round (starting from 1).

(6) MAX_ROUNDS  
The maximum number of allowed rounds (e.g., 3).

------------------------------------------------------------
Your Role
------------------------------------------------------------

You must decide EXACTLY ONE action for the CURRENT round:

1) Answer the question now, OR  
2) Request additional retrieval, OR  
3) (Only if ROUND_INDEX == MAX_ROUNDS) Output a forced final answer.

You are NOT allowed to guess unless forced by the round limit.

------------------------------------------------------------
Core Decision Rules (STRICT)
------------------------------------------------------------

RULE 1 — Answer only if decisive  
You may answer ONLY IF:

- Exactly ONE option is clearly supported by
  GLOBAL_CAPTIONS + RETRIEVED_CONTEXT (if any), AND
- All other options can be reasonably ruled out
  based on the available evidence.

You MUST NOT rely on:
- assumed intent,
- events not described in captions.

------------------------------------------------------------

RULE 2 — Retrieval under uncertainty  
If:
- more than one option remains plausible, OR
- the question depends on brief moments, interruptions,
  ordering, or object states not resolved yet,

AND ROUND_INDEX < MAX_ROUNDS:
→ You SHOULD request retrieval.

------------------------------------------------------------

RULE 3 — Forced convergence at final round  
If ROUND_INDEX == MAX_ROUNDS:

- You MUST NOT request retrieval.
- You MUST output an answer.

In this case:
- Select the option MOST CONSISTENT with the available evidence.
- Do NOT invent new evidence.
- Explicitly ground reasoning in what IS available.

------------------------------------------------------------
Retrieval Request Requirements
------------------------------------------------------------

When requesting retrieval:

- Output ONE concise English sentence.
- The retrieval_query MUST:
  • describe the missing visual evidence,
  • reference the QUESTION,
  • include OPTIONS in parentheses.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

Case 1 — Answer now
{
  "decision": "answer",
  "answer": "<exact option text>",
  "reasoning":
    "<concise explanation grounded in GLOBAL_CAPTIONS
     and RETRIEVED_CONTEXT>"
}

Case 2 — Need retrieval (ROUND_INDEX < MAX_ROUNDS)
{
  "decision": "need_retrieval",
  "retrieval_query":
    "<one-sentence description of missing visual evidence
     with question and options in parentheses>"
}

Case 3 — Forced answer (ROUND_INDEX == MAX_ROUNDS)
{
  "decision": "forced_answer",
  "answer": "<exact option text>",
  "reasoning":
    "Based on the available captions and retrieved context,
     this option is the most consistent, although some ambiguity remains."
}

------------------------------------------------------------
Important Notes
------------------------------------------------------------

- GLOBAL_CAPTIONS are summaries, not exhaustive logs.
- Absence of evidence is NOT evidence of absence.
- Prefer retrieval over guessing UNTIL forced by MAX_ROUNDS.
- Output ONLY the JSON object, nothing else.

============================================================
In-Context Examples (EgoSchema-Aligned)
============================================================

------------------------------------------------------------
Example 1 — Answerable from GLOBAL_CAPTIONS (Round < MAX)
------------------------------------------------------------

QUESTION:
What is the primary activity that occurs multiple times in the video?

OPTIONS:
A. Cooking food on a stove  
B. Folding and shaping dough repeatedly  
C. Cleaning kitchen surfaces  
D. Talking on the phone  

GLOBAL_CAPTIONS (excerpt):
- “Across several segments, I repeatedly divide dough, roll it flat, and shape it into similar pieces.”
- “The same dough-handling actions continue throughout the video with little variation.”

OUTPUT:
{
  "decision": "answer",
  "answer": "B. Folding and shaping dough repeatedly",
  "reasoning": "The global captions consistently describe repeated dough division, rolling, and shaping across multiple segments, while no other listed activities recur."
}

------------------------------------------------------------
Example 2 — Need Retrieval (Round < MAX, Evidence Insufficient)
------------------------------------------------------------

QUESTION:
Although the video is predominantly focused on one recurring action, there is an interruption in the activity. Briefly describe this interruption.

OPTIONS:
A. A phone call interrupts the activity  
B. The person stops to drink water  
C. Another person enters and speaks  
D. The activity pauses to adjust equipment  

GLOBAL_CAPTIONS (excerpt):
- “I repeatedly perform the same hand motion on an object.”
- “There is a short pause before the activity resumes.”

OUTPUT:
{
  "decision": "need_retrieval",
  "retrieval_query": "What happens when my recurring hand action on an object is interrupted, such as stopping to handle another object, pausing to drink, another person entering and interacting, or pausing to adjust equipment?"
(Options: A. A phone call interrupts the activity; B. The person stops to drink water; C. Another person enters and speaks; D. The activity pauses to adjust equipment)
}

------------------------------------------------------------
Example 3 — Forced Answer (ROUND_INDEX == MAX_ROUNDS)
------------------------------------------------------------

QUESTION:
What could help in case of fire?

OPTIONS:
A. A fire extinguisher  
B. A sink with running water  
C. A first aid kit  
D. An open window  

GLOBAL_CAPTIONS (excerpt):
- “I walk through a hallway and pass several wall-mounted objects.”
- “Safety-related equipment appears briefly along the wall.”

RETRIEVED_CONTEXT (from earlier rounds):
- “A red cylindrical object with a hose is mounted on the wall in one segment.”

OUTPUT:
{
  "decision": "forced_answer",
  "answer": "A. A fire extinguisher",
  "reasoning": "The retrieved context mentions a red cylindrical object with a hose mounted on the wall, which is most consistent with a fire extinguisher, while no evidence supports the other options."
}
"""

EGOSCHEMA_PROMPTS["DEFAULT_ENTITY_TYPES"] = ["person", "location", "object", "event"]
EGOSCHEMA_PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
EGOSCHEMA_PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
EGOSCHEMA_PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
EGOSCHEMA_PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
EGOSCHEMA_PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
EGOSCHEMA_PROMPTS["default_text_separator"] = [
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


EGOSCHEMA_PROMPTS["caption_reconstruction"] = """
You are an egocentric episodic caption rewriter
designed for the EgoSchema benchmark.

This prompt is used in the RETRIEVAL STAGE
to selectively regenerate a more precise,
evidence-focused caption from a short egocentric video segment.

------------------------------------------------------------
Goal
------------------------------------------------------------

Given retrieval keywords and a short egocentric video segment,
decide WHETHER the segment contains ANY visual evidence
related to the keywords.

- If NO keyword-relevant visual evidence is present:
  → Output an empty string: ""

- If YES (at least one keyword is visually grounded):
  → Output a rewritten caption that highlights
    keyword-relevant objects, actions, or states,
    while also preserving other clearly visible context.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

• Retrieval keywords derived from the user question
  (objects, actions, states, environments).

• A short egocentric video segment (~5-10 seconds),
  sampled frames in temporal order.

• An ORIGINAL fine-grained caption for the same segment
  (generated earlier and grounded in the frames).

------------------------------------------------------------
CRITICAL RELEVANCE CHECK (MANDATORY)
------------------------------------------------------------

Before writing anything, you MUST perform this check:

Determine whether the current video frames contain
ANY visually observable element that matches
ANY of the retrieval keywords, including:

- a keyword object (or a clear visual instance of it),
- a keyword-related action (e.g., picking up, walking, placing),
- a keyword-related state or spatial relation
  (e.g., near me, in front of me, on a table, moving closer/farther),
- a concrete visual proxy of a keyword
  (e.g., weather cues for "season", tools for "cooking").

If NONE of the above are visible in the frames:
→ Output exactly:
""

DO NOT describe the scene.
DO NOT summarize.
DO NOT rewrite the caption.

------------------------------------------------------------
When Rewriting IS Allowed
------------------------------------------------------------

You MAY rewrite the caption ONLY IF:

- At least ONE retrieval keyword
  is directly grounded in the visible frames.

------------------------------------------------------------
How to Rewrite the Caption
------------------------------------------------------------

If rewriting is allowed:

• Re-examine the frames with emphasis on:
  - keyword-related objects, actions, states, or locations,
  - details that may have been missing or under-specified
    in the original caption.

• Use the ORIGINAL fine-grained caption ONLY as reference to:
  - keep object names consistent,
  - preserve already-correct observations.

You MUST:
- add details ONLY if clearly visible,
- prefer frames over the original caption if they conflict.

You MUST NOT:
- copy the original caption verbatim,
- introduce unseen objects or actions,
- infer intent, purpose, correctness, or outcome.

------------------------------------------------------------
Caption Writing Requirements
------------------------------------------------------------

Write ONE concise caption that describes what is visible,
including (when applicable):

- what I am doing with my hands or body,
- which objects I interact with and how,
- object or environment states
  (held/placed, open/closed, near/far, in view/out of view),
- continuation, repetition, pause, or change of actions.

Focus on keyword-relevant evidence FIRST,
but also include other clearly visible context
that helps situate the scene.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

If keyword-relevant evidence is present, output EXACTLY:

{
  "caption": "<first-person, factual rewritten caption>"
}

If NO keyword-relevant evidence is present, output EXACTLY:

""

No extra text.
No explanations.
No additional fields.

------------------------------------------------------------
Style Constraints
------------------------------------------------------------

• First-person ("I").
• Factual and observational.
• Short paragraph or a few sentences.
• No speculation, no interpretation, no task labels.

------------------------------------------------------------
Important Notes
------------------------------------------------------------

• This step functions as a FILTER + REFINER.
• Not all retrieved segments should survive.
• Dropping irrelevant segments is CORRECT behavior.
• Accuracy and keyword-grounded evidence
  are more important than coverage.

------------------------------------------------------------
Inputs
------------------------------------------------------------
"""


EGOSCHEMA_PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """
------------------------------------------------------------
- Goal (EgoSchema → Event / Entity Memory Retrieval Query) -
------------------------------------------------------------

Given a first-person (egocentric) multiple-choice question from EgoSchema
and its answer options,
rewrite them into EXACTLY ONE concise English declarative sentence
that can be used to retrieve relevant EVENTS, ENTITIES,
and RELATIONS from an egocentric memory system
(e.g., event-centric knowledge graph, entity records, or caption memory).

The rewritten sentence should describe
ONE OR MORE CONCRETE, LOCALLY OBSERVABLE EVENT PATTERNS
that could support ANY of the answer options.

The output is NOT an answer.
It is an EVENT-ORIENTED RETRIEVAL QUERY.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I").
- Do NOT ask a question.
- Do NOT include explanations or reasoning.
- Do NOT include option letters.

------------------------------------------------------------
Core Retrieval Principle (CRITICAL)
------------------------------------------------------------

This query is used for EVENT / ENTITY / RELATION retrieval,
NOT for visual similarity search.

Therefore, the query MUST:
- explicitly name concrete objects or entities,
- explicitly describe observable actions or interactions,
- explicitly mention object states or spatial relations if relevant,
- describe REPEATED or DISCRIMINATIVE EVENTS
  rather than a global video summary.

Do NOT try to resolve which option is correct.
The goal is to retrieve all potentially relevant events.

------------------------------------------------------------
Event-Centric Requirements (MANDATORY)
------------------------------------------------------------

The rewritten sentence SHOULD include, if implied by the question or options:

- concrete actions I perform
  (e.g., pick up, place, cut, sort, carry, move, fold),
- concrete objects or tools involved
  (e.g., clothes, hamper, peas, knife, bucket, pan),
- object states or relations
  (e.g., held vs placed, inside vs outside, near vs far),
- a repeated, dominant, or interrupted event pattern,
- spatial context when it helps distinguish events
  (e.g., in a room, near a table, between rooms).

If the question is GLOBAL or ABSTRACT,
you MUST translate it into
MULTIPLE CONCRETE EVENT TYPES
that could be individually retrieved from memory.

------------------------------------------------------------
Using Multiple-Choice Options (IMPORTANT)
------------------------------------------------------------

You MUST use the answer options to EXPAND the query.

Specifically:
- extract concrete objects, actions, or relations
  mentioned or implied by each option,
- merge them into ONE event-level description,
- ensure the query covers ALL visually distinct alternatives.

Do NOT:
- include abstract traits (e.g., ambitious, humble),
- include inferred intent or purpose,
- include non-visual concepts.

------------------------------------------------------------
Temporal Wording
------------------------------------------------------------

- Use "repeatedly", "at different moments", or "across the video"
  ONLY if the question asks about dominant patterns or sequences.
- Do NOT invent time ranges.

------------------------------------------------------------
Examples (EgoSchema-Aligned, KG-Friendly)
------------------------------------------------------------

Question:
"What are the main ingredients and tools used during the video?
Options: A. Peas, water, salt, knife; B. Peas, water, salt, fork; C. Peas, water, salt, measuring cup, pan, spoon; D. Peas, water, salt, plate; E. Peas, water, salt, bowl"

Output:
Events where I handle peas, water, and salt while using cooking tools or containers such as a knife, fork, measuring cup, pan, or bowl.

------------------------------------------------------------

Question:
"Although the video is predominantly focused on one recurring action, there is an interruption in my activity.
Options: A. I stop the action to interact with another object; B. I pause briefly and then resume the same action; C. I change location and start a different activity; D. I stop entirely and leave the scene"

Output:
Events where I repeatedly perform the same action and then interact with a different object, pause and resume the action, move to a new location, or stop the action.

------------------------------------------------------------

Question:
"What is the primary sequence of actions performed throughout the video?
Options: A. cooking actions (washing/cutting/stirring); B. cleaning actions (scrubbing/rinsing/wiping); C. assembling actions (aligning/fastening/connecting); D. organizing actions (sorting/folding/stacking); E. moving actions (walking/carrying/placing)"

Output:
Repeated events where I perform actions related to cooking, cleaning, assembling, organizing, or moving objects, such as handling items, using tools, or changing locations.

------------------------------------------------------------

Question:
"Based on the video, which moments are most significant in determining my purpose for engaging with the clothes and the hamper in various rooms?
Options: A. sorting clothes; B. disorganization; C. interacting with another person; D. playful movement"

Output:
Events where I pick up, carry, drop, sort, or place clothes near a hamper in different rooms, possibly interacting with other objects or people.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------

Question: {input_text}

------------------------------------------------------------
Output:
------------------------------------------------------------
"""


EGOSCHEMA_PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """
------------------------------------------------------------
- Goal (EgoSchema → Visual Embedding Retrieval Query) -
------------------------------------------------------------

Given a first-person (egocentric) multiple-choice question from EgoSchema
and its answer options,
rewrite them into EXACTLY ONE concise English declarative sentence
to retrieve visually relevant video segments via visual embeddings.

The rewritten sentence should describe
the FULL SET of concrete visual evidence
that could support ANY of the given answer options.

The output is NOT an answer.
It is a high-recall VISUAL RETRIEVAL QUERY.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I") for the camera wearer.
- Do NOT include explanations, reasoning steps, or multiple sentences.

------------------------------------------------------------
Core Retrieval Principle (CRITICAL)
------------------------------------------------------------

This query is used for VISUAL EMBEDDING RETRIEVAL.

Therefore, it should:
- maximize recall over potentially relevant moments,
- include concrete, visually detectable cues,
- avoid abstract interpretation or early disambiguation.

Do NOT try to decide which option is correct.
Do NOT narrow the query too aggressively.

------------------------------------------------------------
Visual Grounding Requirements (MANDATORY)
------------------------------------------------------------

The rewritten query MUST describe ONLY observable visual content, such as:

- my physical actions (walking, picking up, placing, cutting, stirring),
- interactions with concrete objects (clothes, hamper, tools, containers),
- object states or state changes (held, placed, moved, opened, sorted),
- spatial relations and environments (room, kitchen, outdoors, shelves),
- repeated actions, transitions, or changes across time.

You MUST NOT:
- include abstract traits (e.g., ambitious, humble, emotional),
- infer intent beyond visible actions,
- include purely conceptual or non-visual descriptions.

------------------------------------------------------------
How to Use OPTIONS (IMPORTANT)
------------------------------------------------------------

You MUST incorporate information from ALL options
to EXPAND the retrieval scope.

Specifically:
- extract concrete objects, actions, and environments
  mentioned or implied by the options,
- paraphrase them as visually observable alternatives,
- merge them into ONE broad but concrete visual query.

Do NOT:
- copy option letters,
- list options verbatim,
- include abstract labels without visual grounding.

------------------------------------------------------------
Temporal Scope
------------------------------------------------------------

- Use "throughout the video" ONLY when the question asks about:
  • main activity,
  • dominant process,
  • primary sequence,
  • overarching behavior.

- Otherwise, describe moments or series of moments
  without inventing time constraints.

------------------------------------------------------------
Examples (EgoSchema-Aligned, HIGH-RECALL)
------------------------------------------------------------

Question:
"What are the main ingredients and tools used during the video?
Options: A. Peas, water, salt, knife; B. Peas, water, salt, fork; C. Peas, water, salt, measuring cup, pan, spoon; D. Peas, water, salt, plate; E. Peas, water, salt, bowl"

Output:
Segments where I handle peas, water, and salt while visibly using kitchen tools or containers such as a knife, fork, measuring cup, pan, bowl, or similar utensils.

------------------------------------------------------------

Question:
"Although the video is predominantly focused on one recurring action, there is an interruption in my activity. Which best describes it?
Options: A. I stop the action to interact with another object; B. I pause briefly and then resume the same action; C. I change location and start a different activity; D. I stop entirely and leave the scene"

Output:
Segments where I repeatedly perform the same visible action and then either interact with a different object, briefly pause and resume, move to a new location, or stop the action.

------------------------------------------------------------

Question:
"What is the primary sequence of actions performed throughout the video?
Options: A. cooking actions (washing/cutting/stirring); B. cleaning actions (scrubbing/rinsing/wiping); C. assembling actions (aligning/fastening/connecting); D. organizing actions (sorting/folding/stacking); E. moving actions (walking/carrying/placing)"

Output:
Segments showing the dominant repeated action flow throughout the video, such as cooking (washing, cutting, stirring), cleaning (scrubbing, rinsing, wiping), assembling (aligning, fastening, connecting), organizing (sorting, folding, stacking), or moving items (walking, carrying, placing).

------------------------------------------------------------

Question:
"Based on the video, which moments are most significant in determining my purpose for engaging with the clothes and the hamper in various rooms?
Options: A. sorting clothes; B. disorganization; C. interacting with another person; D. playful movement"

Output:
Segments where I pick up, carry, drop, sort, or organize clothes around a hamper in different rooms, possibly interacting with other objects or moving between locations.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------

Question: {input_text}
------------------------------------------------------------
Output:
"""


EGOSCHEMA_PROMPTS[
    "keywords_extraction"
] = """
------------------------------------------------------------
- Goal (EgoSchema STRICT Keyword Extraction for Visual Filtering) -
------------------------------------------------------------

Given a first-person (egocentric) multiple-choice question from the EgoSchema benchmark,
extract a MINIMAL and PRECISE set of KEYWORDS that will be used to
FILTER visual segments based on DIRECT visual overlap.

These keywords are NOT for reasoning.
They are used ONLY to decide whether a video segment
contains any potentially relevant visual evidence.

------------------------------------------------------------
CRITICAL PRINCIPLE (DO NOT VIOLATE)
------------------------------------------------------------

Each extracted keyword MUST correspond to something that can be:

- directly SEEN in a single video frame, OR
- directly OBSERVED as a concrete physical action or object state.

If a keyword cannot be pointed to visually in a frame,
it MUST NOT be included.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output keywords in English only.
- List keywords separated by commas.
- Do NOT output full sentences.
- Do NOT include explanations, reasoning, or interpretations.
- Do NOT include conclusions or answers.
- Do NOT include option letters (A/B/C/…).

------------------------------------------------------------
Mandatory Use of Multiple-Choice Options (STRICT)
------------------------------------------------------------

If the question provides answer options:

- You MUST extract keywords ONLY from:
  • concrete objects mentioned in the options,
  • concrete physical actions mentioned in the options,
  • concrete environments or visible conditions in the options.

- You MUST NOT extract:
  • abstract labels (e.g., purpose, objective, disorganization),
  • meta-descriptions (e.g., primary, main, sequence),
  • inferred intentions or summaries.

If an option is abstract:
→ extract ONLY its most concrete VISUAL PROXY
(e.g., "sorting clothes" → pick up clothes, place clothes).

------------------------------------------------------------
What You MAY Extract
------------------------------------------------------------

ONLY the following are allowed:

1) Concrete objects  
   (bucket, gloves, bicycle pedal, clothes, hamper, knife)

2) Concrete actions  
   (pick up, place, carry, cut, walk, move, adjust)

3) Concrete spatial or state cues  
   (in hand, on table, near me, moving, stationary)

------------------------------------------------------------
What You MUST NOT Extract
------------------------------------------------------------

You MUST NOT include:

- abstract task names (organizing, cooking, cleaning),
- inferred purpose or intention,
- summary words (sequence, activity, behavior),
- evaluative or interpretive terms.

------------------------------------------------------------
Output Format
------------------------------------------------------------

Comma-separated keywords only. 

######################
- Examples (EgoSchema-Aligned, Option-Grounded) -
######################

Question:
"What are the main ingredients and tools used during the video?
Options: A. Peas, water, salt, knife; B. Peas, water, salt, fork; C. Peas, water, salt, measuring cup, pan, spoon; D. Peas, water, salt, plate; E. Peas, water, salt, bowl"

Output:
peas, water, salt, knife, fork, measuring cup, pan, bowl

------------------------------------------------------------

Question:
"What is the primary sequence of actions performed throughout the video?
Options: A. cooking actions (washing/cutting/stirring); B. cleaning actions (scrubbing/rinsing/wiping); C. assembling actions (aligning/fastening/connecting); D. organizing actions (sorting/folding/stacking); E. moving actions (walking/carrying/placing)"

Output:
cooking, washing, cutting, stirring, cleaning, scrubbing, rinsing, wiping, assembling, aligning, fastening, organizing, sorting, folding, stacking, moving, walking, carrying, placing

------------------------------------------------------------

Question:
"Although the video is predominantly focused on one recurring action, there is an interruption in the activity.
Options: A. interact with another object; B. pause briefly and resume; C. change location; D. stop and leave the scene"

Output:
interruption, interact with object, pause, resume action, change location, leaving scene

------------------------------------------------------------

Question:
"Based on the video, which moments can be considered most significant in determining the main character's purpose for engaging with the clothes and the hamper in the various rooms?
Options: A. sorting clothes; B. disorganization; C. interacting with another person; D. playful movement"

Output:
clothes, hamper in various rooms, picking up clothes, placing clothes, sorting clothes, dropping clothes

------------------------------------------------------------

Question:
"What is the weather like during the opening scene of the video?
Options: A. sunny; B. rainy; C. snowy; D. windy"

Output:
opening scene, sunlight, rain, snow, wind

------------------------------------------------------------

Question:
"What is the primary objective of interacting with the bicycle pedal?
Options: A. repairing; B. adjusting; C. cleaning; D. testing"

Output:
bicycle pedal, rotate pedal, tighten, loosen, wipe pedal

------------------------------------------------------------
- Real Data -
------------------------------------------------------------

Question: {input_text}

------------------------------------------------------------
Output:
------------------------------------------------------------
"""


EGOSCHEMA_PROMPTS[
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
