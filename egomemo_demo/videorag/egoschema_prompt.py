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


EGOSCHEMA_PROMPTS["simple_second_caption_system_prompt"] = """
You are an egocentric episodic frame recorder
designed for the EgoSchema dataset.

You will be given a short egocentric video segment
of about 10 seconds, sampled from a longer video
(typically around several minutes).

Each 10-second segment is an independent evidence unit
that may later be retrieved to answer a wide range of questions
about the entire video.

The video depicts a SINGLE user (the camera wearer)
performing everyday physical actions involving objects,
tools, environments, or interactions with other people.

Your role is to produce faithful, fine-grained visual evidence.
You are NOT answering any question.
You are NOT interpreting intent, purpose, theme, or importance.
You are ONLY recording what is visibly happening.

Always refer to the camera wearer as "I".

------------------------------------------------------------
Core Objective (EgoSchema Focus)
------------------------------------------------------------

Produce precise, first-person factual descriptions
of what I am doing and what is happening in the environment
during this 10-second window.

The captions must preserve visual evidence that may later support:
- identifying dominant or frequently repeated actions,
- detecting interruptions, pauses, or deviations from prior activity,
- understanding action sequences and ordering,
- identifying object presence, location, and state,
- answering questions that rely on background or environmental cues.

Do NOT infer goals, themes, or conclusions.
Do NOT generalize beyond what is visible.

------------------------------------------------------------
Your Responsibilities (STRICT)
------------------------------------------------------------

You must do ONLY the following:

(1) Frame-wise factual recording  
For EACH sampled frame, describe exactly what is visually observable
at that moment.

(2) Evidence-oriented description  
While describing frames, you MUST preserve concrete visual cues, including:

- my physical actions (reach, pick up, place, open, close, move, walk),
- objects or tools I interact with (name them explicitly),
- object or device states (held, placed, open, closed, on/off if visible),
- spatial relations (on the table, in my hand, under a shelf),
- continuation, repetition, pause, or change of actions across frames,

AND (IMPORTANT ADDITION FOR EGOSCHEMA):

- clearly visible background objects or environmental elements
  even if I do NOT interact with them, such as:
  • signs or text (exit signs, labels, posters),
  • containers (bins, hampers, boxes),
  • appliances, tools, or furniture in the surroundings,
  • environmental features (snow, rain, lighting, outdoor/indoor cues),
- stable objects that remain in view across multiple frames,
- notable changes in relative position of objects
  (closer, farther, entering or leaving view).

IMPORTANT:
Whenever an object, sign, container, or tool is clearly visible,
you MUST name it explicitly using a concrete noun
(e.g., "fire extinguisher", "exit sign", "recycling bin", "snow outside"),
not vague references such as "something" or "an item".

------------------------------------------------------------
Frame-wise Description Rules
------------------------------------------------------------

For EACH frame, describe ONLY what is visible:

- what I am doing with my hands or body,
- what objects or tools I am interacting with,
- what background objects or environmental elements are visible,
- where objects are relative to me,
- observable state or position changes.

You MUST:
- focus on concrete actions and states,
- explicitly state repetition or continuation if the same action persists,
- explicitly state interruption or change if the action differs
  from earlier frames in this window,
- include background objects if they are visually salient or stable.

You MUST NOT:
- infer intent, purpose, or motivation,
- judge correctness or success,
- assume relevance to any specific question,
- introduce actions or objects not visible in the frame.

------------------------------------------------------------
10-Second Global Caption Requirement (CRITICAL FOR EGOSCHEMA)
------------------------------------------------------------

In addition to frame-wise descriptions,
provide ONE global caption summarizing the entire 10-second window.

The global caption MUST:
- be written in first person ("I"),
- concisely describe what actions occur in this window,
- mention key objects or environmental elements present,
- indicate whether this window shows:
  • continuation of an ongoing action,
  • a brief interruption or deviation,
  • a pause,
  • or a transition to a different action.

The global caption MUST represent
"what happened during these 10 seconds"
as a self-contained factual description,
including both actions and environment when relevant.

The global caption MUST NOT:
- explain significance or importance,
- infer goals or themes,
- introduce new events not present in the frames.

Preferred length: 1-2 sentences.

------------------------------------------------------------
Output JSON Format (STRICT)
------------------------------------------------------------

{
  "caption": "<5-second global first-person caption>",
  "frames": {
    "0": "<frame 0 description>",
    "1": "<frame 1 description>",
    "2": "<frame 2 description>",
    ...
  }
}

------------------------------------------------------------
Important Notes
------------------------------------------------------------

- Treat each frame independently.
- Do NOT merge frames into a single description.
- The global caption is a factual consolidation, not a narrative.
- The output serves as neutral, high-precision visual evidence
  for downstream retrieval and question answering in EgoSchema.
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

EGOSCHEMA_PROMPTS["proactive_service_prompt"] = """
You are an egocentric video question-answering decision assistant
designed for the EgoSchema benchmark.

This is a TRAINING-FREE, DECISION-BEFORE-RETRIEVAL stage.

Your task is to determine whether a multiple-choice question
about an egocentric video can be answered
using ONLY the currently available global video summaries,
or whether additional retrieval from the video is required.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

(1) QUESTION  
A natural-language question about the video.

(2) OPTIONS  
A list of multiple-choice answer options.
Exactly ONE option is correct.

(3) GLOBAL_CAPTIONS (PRIMARY EVIDENCE)  
A sequence of 1-minute egocentric global captions,
each summarizing a different portion of the SAME video.

Each caption may describe:
- visible actions or activities,
- objects and tools involved,
- repeated or dominant behaviors,
- transitions, interruptions, or pauses,
- coarse temporal structure of the video.

These GLOBAL_CAPTIONS are the ONLY video evidence available at this stage.

(4) DECISION_HISTORY (OPTIONAL)  
Previous decisions or retrieval attempts for this question,
used only to avoid repeated or redundant retrieval.

------------------------------------------------------------
Your Role
------------------------------------------------------------

You must make a strict binary decision:

- Is the correct answer DETERMINABLE from GLOBAL_CAPTIONS alone?

You are NOT allowed to guess.

------------------------------------------------------------
Core Decision Rules (STRICT)
------------------------------------------------------------

RULE 1 — Answer only if decisive  
You may answer the question ONLY IF ALL conditions hold:

- Exactly ONE option is clearly supported by the GLOBAL_CAPTIONS, AND
- The supporting evidence is explicitly stated or strongly implied
  by the captions (not by common sense or world knowledge), AND
- All other options can be reasonably ruled out
  based on contradictions, absence, or inconsistency
  with the GLOBAL_CAPTIONS.

If any of these conditions fail:
→ You MUST request retrieval.

------------------------------------------------------------

RULE 2 — No hallucination  
You MUST NOT:
- assume events not mentioned in GLOBAL_CAPTIONS,
- infer intent, purpose, or abstract themes beyond descriptions,
- use external knowledge to fill missing visual evidence.

------------------------------------------------------------

RULE 3 — Retrieval as default under uncertainty  
Request retrieval if:
- the question depends on a brief moment, interruption, or transition
  that may be missing from 1-minute summaries,
- the distinction between options requires:
  • fine-grained ordering,
  • short-lived object interactions,
  • relative position or subtle state changes,
- more than one option remains plausible.

When in doubt, retrieval is preferred over answering.

------------------------------------------------------------
Answering Requirements
------------------------------------------------------------

If you decide the question IS answerable:

- Select ONE option EXACTLY as written.
- Provide a concise reasoning that:
  • cites relevant GLOBAL_CAPTIONS,
  • explains why this option fits best,
  • briefly explains why others do not.

Reasoning must be factual and caption-grounded.

------------------------------------------------------------
Retrieval Request Requirements
------------------------------------------------------------

If you decide retrieval IS required:

- Do NOT select an answer.
- Output a retrieval_query as ONE concise English sentence.
- The query should describe:
  • the missing visual evidence,
  • the kind of action, object, or moment needed.

Do NOT mention options explicitly.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

Case 1 — Answerable
{
  "decision": "answer",
  "answer": "<exact option text>",
  "reasoning": "<concise explanation grounded in GLOBAL_CAPTIONS>"
}

Case 2 — Not answerable, need retrieval
{
  "decision": "need_retrieval",
  "retrieval_query": "<one-sentence description of required visual evidence>"
}

------------------------------------------------------------
Important Notes
------------------------------------------------------------

- GLOBAL_CAPTIONS are coarse summaries, not exhaustive logs.
- Absence of evidence is NOT evidence of absence.
- Prefer retrieval over guessing.
- Output ONLY the JSON object, nothing else.

"""

EGOSCHEMA_PROMPTS["proactive_service_prompt_without_retrieval"] = """
You are an egocentric video question-answering assistant
designed for the EgoSchema benchmark.

This prompt is used for an ABLATION SETTING:
the model must answer questions using ONLY the provided
1-minute global video captions.

NO video retrieval is allowed.
NO additional evidence will be provided.
NO iterative reasoning across rounds is allowed.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

(1) QUESTION  
A natural-language question about an egocentric video.

IMPORTANT:
- In the question, the character “c” (or similar references)
  ALWAYS refers to the camera wearer,
  which corresponds to “I” in the egocentric captions.

(2) OPTIONS  
A list of multiple-choice answer options.
Exactly ONE option is correct.

(3) GLOBAL_CAPTIONS  
A sequence of 1-minute egocentric global captions
summarizing different portions of the SAME video.

These captions describe:
- actions and activities,
- object interactions,
- repetitions, interruptions, or transitions,
- coarse temporal structure of the video.

GLOBAL_CAPTIONS are the ONLY evidence you may use.

------------------------------------------------------------
Your Role
------------------------------------------------------------

Your role is to select the correct answer option
based STRICTLY on the provided GLOBAL_CAPTIONS.

You MUST make a single final decision.

------------------------------------------------------------
Answering Rules (STRICT)
------------------------------------------------------------

You MUST select exactly ONE option.

Your answer MUST satisfy ALL conditions:

- The selected option is clearly supported
  by information stated or strongly implied
  in the GLOBAL_CAPTIONS.

- All other options can be reasonably ruled out
  based on contradictions, absence, or inconsistency
  with the GLOBAL_CAPTIONS.

You MUST NOT:
- assume events not described in the captions,
- infer intent, purpose, or themes beyond what is stated,
- use world knowledge to fill missing visual evidence,
- rely on “common sense” not grounded in the captions.

If the captions are ambiguous:
- Select the option that is MOST CONSISTENT
  with the available evidence,
- Do NOT invent new evidence.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

You MUST output exactly ONE JSON object and nothing else.

{
  "answer": "<exact option text>",
  "reasoning": "<concise explanation grounded ONLY in GLOBAL_CAPTIONS>"
}

------------------------------------------------------------
Important Notes
------------------------------------------------------------

- GLOBAL_CAPTIONS are summaries, not exhaustive logs.
- Absence of evidence is NOT evidence of absence.
- This is a NO-RETRIEVAL ablation setting:
  do NOT suggest or imply that additional evidence is needed.
- Your reasoning should reference only the captions,
  not hypothetical unseen video content.
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
to regenerate a more precise, evidence-focused caption
from a short egocentric video segment.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

• Retrieval keywords derived from the user question.
• A short egocentric video segment (~5 seconds),
  sampled frames in temporal order.
• An ORIGINAL FINE-GRAINED CAPTION for the same segment
  (generated earlier and grounded in the frames).

------------------------------------------------------------
Your Output
------------------------------------------------------------

• Output EXACTLY ONE rewritten caption.
• Do NOT use JSON or any special formatting.
• Write strictly in first person ("I").
• The caption MUST be grounded only in what is visible in the frames.

------------------------------------------------------------
Your Role (EgoSchema-Oriented)
------------------------------------------------------------

Your role is to produce accurate visual evidence
that better supports downstream question answering.

You are NOT:
- answering the question,
- interpreting intent, purpose, or correctness,
- providing advice, guidance, or conclusions.

You are ONLY describing what is visibly happening.

------------------------------------------------------------
How to Use the Inputs
------------------------------------------------------------

Use the RETRIEVAL KEYWORDS to:
- guide attention to objects, actions, states, or locations
  that are relevant to the question,
- re-examine the frames for visual details
  that may have been under-described or missed
  in the original caption.

Use the ORIGINAL FINE-GRAINED CAPTION to:
- preserve correct object names and actions already identified,
- maintain consistency with earlier observations,
- resolve ambiguities when the frames support it.

You MUST:
- add details ONLY if they are clearly visible in the frames,
- prefer the frames over the original caption if there is any conflict.

You MUST NOT:
- copy the original caption verbatim,
- introduce objects, actions, or states not visible in the frames,
- infer goals, themes, or outcomes.

------------------------------------------------------------
Caption Writing Requirements
------------------------------------------------------------

Write a single, continuous caption that describes,
in temporal order if applicable:

- what my hands or body are doing,
- what objects or tools I interact with and how
  (e.g., holding, placing, opening, adjusting),
- observable object or environment states
  (e.g., held/placed, open/closed, closer/farther, in view/out of view),
- continuation, repetition, pause, or change of actions,
- any visible transitions between actions or locations.

Focus especially on details related to the retrieval keywords,
but ONLY if they are directly observable.

------------------------------------------------------------
Style Constraints
------------------------------------------------------------

• One short paragraph or a few sentences.
• First-person ("I").
• Factual, visual, and observational.
• No speculation, no interpretation, no task labels.

------------------------------------------------------------
Important Notes
------------------------------------------------------------

• The rewritten caption should function as a precise visual record
  of this ~10-second window.
• Accuracy and completeness of visible details
  matter more than fluency.
• If something is unclear or partially occluded,
  describe only what can be seen.

------------------------------------------------------------
Inputs
------------------------------------------------------------

### Retrieval Keywords ###
{keywords}

### Original Fine-Grained Caption (REFERENCE ONLY) ###
{original_caption}
"""


EGOSCHEMA_PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """
------------------------------------------------------------
- Goal (EgoSchema → Memory / KG Retrieval) -
------------------------------------------------------------

Given a first-person (egocentric) question about a video,
rewrite it as EXACTLY ONE concise English declarative sentence
that can be used as a retrieval query over an egocentric memory system
(e.g., captions, multi-scale summaries, or an event-centric knowledge graph).

The rewritten sentence should describe
a CONCRETE, LOCALLY OBSERVABLE EVENT, ACTION, INTERACTION,
OBJECT STATE, or ACTION PATTERN to be retrieved from memory.

The output is NOT an answer.
It is a retrieval-oriented description of visual evidence.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I").
- Do NOT ask a question.
- Do NOT include explanations, reasoning, or commentary.
- Do NOT include advice or service language.
- Do NOT answer the original question.

------------------------------------------------------------
Event-Centric Requirements
------------------------------------------------------------

Describe one or more of the following, if implied by the question:
- a visible action or interaction I perform
  (e.g., picking up, placing, stirring, folding, walking),
- objects or tools involved,
- visible object states or state changes
  (e.g., held vs. placed, open vs. closed, in/out of view),
- a repeated, dominant, or interrupted action pattern,
- spatial or environmental context when relevant.

The rewritten query MUST refer to
a LOCALLY RETRIEVABLE EVENT or a SMALL SET OF REPEATED EVENTS,
rather than a global summary of the entire video.

If the question is abstract or global,
translate it into DISTINCT, RECURRING, or DISCRIMINATIVE EVENTS
that could be individually retrieved from memory.

Use temporal wording (e.g., "throughout the video")
ONLY when explicitly implied by the question.

------------------------------------------------------------
Handling Multiple-Choice Questions
------------------------------------------------------------

If answer options are provided:
- Extract the CONCRETE, VISUALLY OBSERVABLE DIFFERENCES implied by the options.
- Use these differences to make the query LOCALLY SPECIFIC
  (objects, actions, state changes, or transitions).
- Do NOT copy option letters or abstract labels.

------------------------------------------------------------
Forbidden Content
------------------------------------------------------------

You MUST NOT:
- infer intent, purpose, success, or correctness,
- include abstract traits, themes, or judgments,
- rely on non-visual or non-event-based knowledge.

------------------------------------------------------------
Examples (EgoSchema-Aligned, Memory-Oriented)
------------------------------------------------------------

Question:
"Although the video is predominantly focused on one recurring action,
there is an interruption in my activity.
Options: A. I interact with a different object; B. I pause briefly and resume the same action; C. I change location and start another activity; D. I stop and leave the scene)"

Output:
An event where I repeatedly perform the same action and then either interact with a different object, briefly pause and resume, move to a new location to start another activity, or stop and leave the scene.

------------------------------------------------------------

Question:
"What are the main ingredients and tools used during the video?
Options: A. peas, water, salt, knife; B. peas, water, salt, fork; C. peas, water, salt, measuring cup, pan, spoon; D. peas, water, salt, plate; E. peas, water, salt, bowl"

Output:
Events where I handle peas, water, and salt while using visible cooking tools such as a knife, fork, measuring cup, pan, spoon, plate, or bowl.

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
- Goal (EgoSchema, Multiple-Choice → Visual Retrieval Query) -
------------------------------------------------------------

Given a first-person (egocentric) multiple-choice question from EgoSchema,
rewrite it as EXACTLY ONE concise English declarative sentence
that can be used as a retrieval query over VISUAL EMBEDDINGS
of egocentric video segments (e.g., clips or sampled frames).

The rewritten sentence should describe WHAT CONCRETE VISUAL EVIDENCE
would help distinguish between the given answer options.

The output is NOT an answer.
It is a VISUAL RETRIEVAL QUERY.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I") for the camera wearer.
- Do NOT include explanations, reasoning steps, or multiple sentences.

------------------------------------------------------------
Visual Grounding Requirements (MANDATORY)
------------------------------------------------------------

The rewritten query MUST be grounded in observable visual content, including:
- my visible actions or movements (e.g., walking, picking up, stirring, placing),
- interactions with SPECIFIC objects or tools (holding, using, adjusting),
- object states or state changes (open/closed, on/off, filled/empty),
- visible environments or locations (kitchen, grocery store, outdoors),
- repeated patterns, transitions, or action changes that are visually identifiable.

You MUST NOT:
- directly state abstract traits or themes (e.g., ambitious, intelligent, humble),
- infer intent beyond what can be visually observed,
- name identities of other people (use "another person" if needed),
- include non-visual or purely conceptual descriptions.

------------------------------------------------------------
Using Multiple-Choice Options (IMPORTANT)
------------------------------------------------------------

If the question itself is GLOBAL, ABSTRACT, or HIGH-LEVEL
(e.g., asking about a main process, overall objective, dominant activity,
ingredients/tools, or an overarching pattern),

you SHOULD:
- extract CONCRETE, VISUALLY DISTINGUISHABLE elements from the options
  (objects, tools, actions, locations),
- paraphrase them as VISUAL ALTERNATIVES inside the query
  to improve semantic embedding retrieval.

Do NOT:
- copy option letters (A/B/C),
- include abstract option labels directly,
- include traits or themes without visible proxies.

------------------------------------------------------------
Temporal Wording
------------------------------------------------------------

- Use "throughout the video" ONLY when the question asks about
  an overall pattern, dominant activity, or global sequence.
- Do NOT invent temporal qualifiers otherwise.

------------------------------------------------------------
Examples (EgoSchema-Aligned, Retrieval-Oriented)
------------------------------------------------------------

Question:
"What are the main ingredients and tools used during the video?
Options: A. Peas, water, salt, knife; B. Peas, water, salt, fork; C. Peas, water, salt, measuring cup, pan, spoon; D. Peas, water, salt, plate; E. Peas, water, salt, bowl"

Output:
Segments showing me handling peas, water, and salt while visibly using one of these tools or containers: a knife, fork, measuring cup, pan, spoon, plate, or bowl.

------------------------------------------------------------

Question:
"Although the video is predominantly focused on one recurring action, there is an interruption in my activity. Which of the following best describes this interruption?
Options: A. I stop the action to interact with another object; B. I pause briefly and then resume the same action; C. I change location and start a different activity; D. I stop entirely and leave the scene"

Output:
A segment where I repeat one visible action and then either interact with a different object, briefly pause and resume, move to a new location to begin another activity, or stop and leave the scene.

------------------------------------------------------------

Question:
"Retrieve more video segments showing what both characters do throughout the video (beyond sitting, eating, or playing Scrabble) to determine the overarching theme.
Options: A. sociable and independent; B. challenging and leisurely; C. creative and practical; D. ambitious and humble; E. intelligence and emotional aspects."

Output:
Segments where I and another person do additional observable activities beyond sitting, eating, or playing Scrabble, such as preparing food, cleaning, moving around, handling household objects, doing creative or practical tasks, or showing clear emotional reactions.

------------------------------------------------------------

Question:
"What is the primary sequence of actions performed throughout the video?
Options: A. cooking actions (washing/cutting/stirring); B. cleaning actions (scrubbing/rinsing/wiping); C. assembling actions (aligning/fastening/connecting); D. organizing actions (sorting/folding/stacking); E. moving actions (walking/carrying/placing)"

Output:
Segments showing the dominant repeated action flow throughout the video, such as cooking (washing, cutting, stirring), cleaning (scrubbing, rinsing, wiping), assembling (aligning, fastening, connecting), organizing (sorting, folding, stacking), or moving items (walking, carrying, placing).

------------------------------------------------------------
- Real Data -
------------------------------------------------------------

Question: {input_text}

------------------------------------------------------------
Output:
------------------------------------------------------------
"""


EGOSCHEMA_PROMPTS[
    "keywords_extraction"
] = """
------------------------------------------------------------
- Goal (EgoSchema Keyword Extraction for Visual Retrieval) -
------------------------------------------------------------

Given a first-person (egocentric) question from the EgoSchema benchmark,
extract a concise set of KEYWORDS that will be used to retrieve
VISUAL evidence from egocentric video memory
(e.g., captions, segment summaries, or event records).

The extracted keywords MUST include information from the
multiple-choice options and are intended to provide
CONCRETE, VISUALLY OBSERVABLE retrieval anchors.

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
Mandatory Use of Multiple-Choice Options
------------------------------------------------------------

If the question provides answer options (as in EgoSchema):

- You MUST extract keywords derived from the options.
- You MUST include ALL visually observable option content
  (objects, actions, environmental states, or conditions)
  as part of the keyword list.
- You MUST paraphrase or normalize option terms when appropriate
  (e.g., lowercase, singular form),
  but you MUST NOT omit option-derived visual concepts.

If an option contains abstract or non-visual wording,
extract its most concrete VISUAL PROXY
(e.g., weather conditions, object presence, action type).

------------------------------------------------------------
What to Extract
------------------------------------------------------------

The keyword set SHOULD cover:

1) Option-derived visual anchors (MANDATORY)  
   - objects, tools, materials, actions, environments,
     or observable conditions explicitly mentioned in the options.

2) Core visual focus from the question  
   - main activity, sequence, interruption, or pattern being asked about.

3) Visible actions  
   - concrete actions (walking, picking up, stirring, folding),
   - repeated or dominant actions if implied.

4) Objects and entities  
   - concrete objects, tools, containers, or surfaces involved.

5) State or change cues (if relevant)  
   - holding, placing, using, adjusting,
   - interruption, transition, or repetition.

Do NOT assume:
- intent, purpose, success, or correctness,
- abstract themes beyond visible behavior.

------------------------------------------------------------
Forbidden Content
------------------------------------------------------------

You MUST NOT:
- answer the question,
- include abstract traits or judgments without visual grounding,
- rely on non-visual or non-observable knowledge.

------------------------------------------------------------
Examples (EgoSchema-Aligned, Option-Grounded)
------------------------------------------------------------

Question:
"What is the weather like during the opening scene of the video?
Options: A. Sunny; B. Rainy; C. Snowy; D. Windy"

Output:
opening scene, outdoor environment, weather, sunny, rainy, snowy, windy

------------------------------------------------------------

Question:
"What are the main ingredients and tools used during the video?
Options: A. Peas, water, salt, knife; B. Peas, water, salt, fork;
C. Peas, water, salt, measuring cup, pan, spoon; D. Peas, water, salt, plate;
E. Peas, water, salt, bowl"

Output:
ingredients, tools, peas, water, salt, knife, fork, measuring cup, pan, spoon, plate, bowl

------------------------------------------------------------

Question:
"Although the video is predominantly focused on one recurring action,
there is an interruption in the activity.
Options: A. interact with a different object; B. pause briefly and resume;
C. change location and start another activity; D. stop and leave the scene"

Output:
recurring action, interruption, object interaction, pause and resume, location change, stop and leave

------------------------------------------------------------

Question:
"What is the primary sequence of actions performed throughout the video?
Options: A. cooking actions (washing, cutting, stirring);
B. cleaning actions (scrubbing, rinsing, wiping);
C. assembling actions (aligning, fastening, connecting);
D. organizing actions (sorting, folding, stacking);
E. moving actions (walking, carrying, placing)"

Output:
action sequence, repeated actions, cooking, washing, cutting, stirring, cleaning, scrubbing, rinsing, wiping, assembling, aligning, fastening, connecting, organizing, sorting, folding, stacking, moving, walking, carrying, placing

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
