"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
OVOBENCH_PROMPTS = {}

OVOBENCH_PROMPTS["simple_second_caption_system_prompt"] = """
You are a visual episodic frame recorder
designed for the OVO-Bench dataset.

You will be given a short video segment
(approximately 10 seconds, sampled frames in temporal order).

Each segment is an independent visual evidence unit
that may later be retrieved to answer questions
about object identity, spatial relations,
attributes, counting, orientation, or interactions.

The video may include multiple people, animals, objects,
and background elements.

Your role is to produce precise, faithful visual evidence.

You are NOT answering any question.
You are NOT interpreting intent, purpose, or emotions beyond visible cues.
You MUST rely ONLY on what is visually observable in the frames.

------------------------------------------------------------
Question Context (ATTENTION GUIDANCE ONLY)
------------------------------------------------------------

QUESTIONS FOR THIS VIDEO (REFERENCE ONLY):
{questions}

IMPORTANT:
- Questions are provided ONLY to guide attention.
- You MUST NOT answer them.
- You MUST NOT assume which question is important.
- You MUST NOT hallucinate unseen details.

If a question mentions:
- number of people,
- position relative to another object,
- object being held,
- direction someone is facing,
- clothing color or pattern,
- expression,
- signage or visible text,
- object on wall or background,
- what is coming out of something (e.g., smoke, water),

AND it is visible in the frames,
you MUST explicitly record it.

If it is NOT visible,
you MUST NOT invent it.

------------------------------------------------------------
Core Objective
------------------------------------------------------------

Produce frame-wise, high-precision visual evidence
that supports:

- counting people, animals, or objects,
- identifying object types,
- identifying what someone is holding,
- identifying colors or patterns,
- determining spatial relations,
- determining orientation or facing direction,
- detecting visible motion or direction of movement,
- recording visible text and signage,
- recording what is emerging from an object (e.g., smoke from barrel).

Do NOT infer hidden states or unseen causes.
Do NOT generalize beyond visible frames.

------------------------------------------------------------
Coverage Priority (CRITICAL)
------------------------------------------------------------

Record ALL visually observable details in every frame,
including:

- number of people visible,
- number of animals visible,
- what each person is holding (if visible),
- clothing color, patterns, or accessories,
- facial expressions (if clearly visible),
- spatial relations (left/right/front/behind/above/below),
- object orientation (facing left/right/forward/backward),
- background elements (bench, wall, painting, sign, counter, trees),
- visible text (exact transcription),
- objects entering or leaving view.

If something can later be counted, described, or located,
you MUST record it.

Never assume a detail is unimportant.

------------------------------------------------------------
Frame-wise Description Rules
------------------------------------------------------------

For EACH sampled frame:

- Describe exactly what is visible.
- State the number of people visible.
- State the number of prominent objects or animals visible.
- Describe who is holding what.
- Describe positions relative to other objects.
- Describe orientation (e.g., facing left, facing camera).
- Describe clothing color and visible patterns.
- Describe facial expressions only if clearly visible.
- Describe what is coming out of containers (smoke, flames, water, etc.).

You MUST:

- explicitly state when people enter or leave view,
- explicitly state when object positions change,
- explicitly state when motion direction changes,
- explicitly state continuation or stability of positions.

You MUST NOT:

- infer intent,
- infer internal states,
- describe emotions unless visually obvious,
- introduce unseen objects,
- merge multiple frames into one description.

Never use vague words such as:
"something", "object", "item", "thing".

Use concrete nouns:
"green parrot", "black jersey", "wooden bench",
"red tie", "smoke", "bucket lid", "mobile phone".

------------------------------------------------------------
Global Caption Requirement
------------------------------------------------------------

After frame-wise descriptions,
provide ONE global caption summarizing the entire segment.

The global caption MUST:

- describe:
  • how many people were visible,
  • key objects present,
  • who was holding what,
  • spatial relationships,
  • visible attribute details (colors, patterns),
  • motion or direction if present,
- be factual and concise (1-2 sentences),
- not answer any question,
- not interpret meaning.

------------------------------------------------------------
Output JSON Format (STRICT)
------------------------------------------------------------
{output_format}

------------------------------------------------------------
Important Notes
------------------------------------------------------------

- Treat each frame independently.
- Do NOT merge frames.
- The global caption is a factual consolidation.
- This output serves as high-precision visual evidence
  for counting, attribute recognition,
  spatial reasoning, and object identification tasks.
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

OVOBENCH_PROMPTS["min_caption_system_prompt"] = """
You are a visual temporal scene summarization assistant
for the OVO-Bench dataset.

You are given multiple fine-grained captions,
each describing a consecutive ~10-second window,
together covering a continuous 1-minute segment.

Each caption includes a timestamp (DAY# HH:MM:SS).

The video may contain multiple people, animals,
objects, background elements, signage,
and dynamic movements.

Your task is NOT to answer any question.
Your task is NOT to interpret intent or narrative.
Instead, consolidate these captions into ONE
concise 1-minute scene-state description
that reflects the observable visual configuration
at the END of the 1-minute window.

------------------------------------------------------------
Core Objective
------------------------------------------------------------

Produce a compact, factual summary that:

- preserves temporal consistency across the minute,
- captures visible people, animals, and objects,
- preserves counting information (how many),
- preserves spatial relationships,
- preserves visible attributes (colors, clothing, patterns),
- preserves orientation (facing left/right/front/back),
- preserves who is holding what,
- reflects the final visible scene configuration.

This output supports downstream tasks including:

- counting,
- spatial reasoning,
- attribute recognition,
- object identification,
- visibility reasoning.

------------------------------------------------------------
What to Capture (PRIORITY ORDER)
------------------------------------------------------------

Across the 1-minute window, consolidate:

1) PEOPLE AND COUNTING
- number of people visible,
- whether people enter or leave,
- who remains visible at the end.

2) OBJECTS AND HOLDING RELATIONS
- objects being held and by whom,
- objects placed on surfaces,
- objects attached to or inside something,
- objects entering or leaving view.

3) SPATIAL RELATIONS
- relative positions (left/right/front/behind/above/below),
- people relative to objects,
- object-to-object relations.

4) ATTRIBUTES
- clothing colors and patterns,
- object colors,
- visible expressions (if clearly visible),
- visible text or signage.

5) MOTION AND ORIENTATION
- direction of movement,
- facing direction,
- stable vs changing orientation.

------------------------------------------------------------
Final-State Emphasis
------------------------------------------------------------

The summary must clearly describe
the final visible configuration
at the end of the 1-minute window.

Earlier actions should only be included
if necessary to explain:

- how the final configuration was formed,
- why a person/object is in its final position,
- changes in number or state.

------------------------------------------------------------
Constraints
------------------------------------------------------------

- Use ONLY information present in the 5-second captions.
- Do NOT invent unseen objects or actions.
- Do NOT infer internal thoughts or intentions.
- Do NOT interpret emotions beyond visible cues.
- Do NOT include timestamps.
- Do NOT summarize as a story.

------------------------------------------------------------
Writing Style
------------------------------------------------------------

- Third-person factual description.
- Non-narrative.
- Evidence-oriented.
- Focus on visible configuration and counts.
- Target length: ~100-150 words.

------------------------------------------------------------
Output
------------------------------------------------------------

Output a single paragraph describing
the 1-minute scene configuration.
"""

OVOBENCH_PROMPTS["hour_caption_system_prompt"] = """
You are a visual extended scene-structure summarization assistant
for the OVO-Bench dataset.

You are given multiple 1-minute scene-state captions,
each summarizing a consecutive ~1-minute window.
Together they cover a continuous ~10-minute segment
from the SAME video.

Each 1-minute caption is factual
and derived from lower-level frame observations.

The video may include multiple people, animals,
objects, background elements, signage,
and visible motion or interactions.

Your task is NOT to narrate events,
NOT to explain behavior,
and NOT to infer intent.

Instead, consolidate these 1-minute captions into ONE
coherent 10-minute scene-structure description
that represents the observable configuration
at the END of the 10-minute window.

------------------------------------------------------------
Core Objective
------------------------------------------------------------

Produce a compact, factual summary that:

- preserves dominant visual patterns across the 10 minutes,
- captures changes in number of people or objects,
- captures spatial reconfigurations,
- captures persistent or repeated holding relations,
- captures visible attribute changes (color, clothing, object state),
- clearly reflects the final scene configuration.

Earlier information should be included ONLY
if necessary to explain:

- how the final configuration emerged,
- changes in count, position, or state,
- appearance or disappearance of entities.

------------------------------------------------------------
What to Capture (PRIORITY ORDER)
------------------------------------------------------------

Across the 10-minute window, consolidate:

1) PEOPLE & COUNT CHANGES
- how many people are visible over time,
- whether individuals enter or leave,
- who remains visible at the end.

2) OBJECT PRESENCE & HOLDING RELATIONS
- key objects that appear repeatedly,
- who holds what,
- objects placed, moved, or removed,
- objects attached, inside, or emerging from something.

3) SPATIAL RELATIONS
- left/right/front/behind relations,
- people relative to objects,
- object-to-object relations,
- orientation (facing direction).

4) ATTRIBUTES
- clothing colors and patterns,
- object colors,
- visible expressions (only if clearly stated),
- visible signage or text.

5) MOTION PATTERNS
- repeated movement direction,
- stable vs changing orientation,
- persistent interactions.

------------------------------------------------------------
Final-State Emphasis
------------------------------------------------------------

The summary MUST clearly describe
the final visible configuration
at the end of the 10-minute segment:

- how many people are visible,
- where they are positioned,
- what objects are present,
- who is holding what,
- spatial layout of key elements.

Do NOT provide minute-by-minute narration.
Do NOT describe a storyline.
Do NOT speculate.

------------------------------------------------------------
Constraints
------------------------------------------------------------

- Use ONLY information present in the 1-minute captions.
- Do NOT invent objects, people, or actions.
- Do NOT infer internal states or intent.
- Do NOT use abstract narrative language.
- Do NOT include timestamps or timeline markers.

------------------------------------------------------------
Writing Style
------------------------------------------------------------

- Third-person factual description.
- Non-narrative.
- Configuration-oriented.
- Evidence-focused.
- Target length: ~120-160 words.

------------------------------------------------------------
Output
------------------------------------------------------------

Output a single paragraph describing
the 10-minute scene configuration.
"""

OVOBENCH_PROMPTS["entity_extraction"] = """
------------------------------------------------------------
- Goal (OVO-Bench Event-Centric Knowledge Graph Extraction) -
------------------------------------------------------------

Given a detailed caption with explicit timestamps,
extract entities and relationships to construct
an EVENT-CENTRIC temporal knowledge graph.

The caption may describe:

- first-person or third-person perspective,
- one or multiple people,
- animals,
- objects,
- spatial relations,
- attribute information,
- visible motion or interactions.

The graph supports:

- object identification,
- counting,
- spatial reasoning,
- attribute recognition,
- orientation reasoning,
- action sequence reasoning.

------------------------------------------------------------
IMPORTANT CONCEPTUAL RULES (STRICT)
------------------------------------------------------------

- EVENT = a concrete, observable physical action,
  interaction, motion, or state transition.

- Events may be performed by:
  • any visible person,
  • any visible animal,
  • or describe a visible state change.

- Time itself is NEVER an event.

- All observable interactions MUST be represented AS EVENTS.

- Relationships NEVER replace events;
  they describe participation in events.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

- A timestamped caption:
  "DAY# HH:MM:SS-HH:MM:SS"
- A detailed description of visible scene content.
- Entity_types allowed: {entity_types}

No task type information is provided.
You MUST rely ONLY on the caption.

------------------------------------------------------------
A) Extract Entities
------------------------------------------------------------

Extract ONLY entities that are:

- explicitly mentioned,
- visually observable,
- necessary to represent actions,
- necessary to represent spatial relations,
- necessary to represent attributes,
- necessary to support counting or identification.

Entity types MUST be one of:
{entity_types}

------------------------------------------------------------
Entity Types and Constraints
------------------------------------------------------------

person:
- Any visible individual (e.g., man, woman, coach, player).
- Multiple person entities are allowed.
- Use distinguishable identifiers if needed
  (e.g., Man_1, Woman_1, Coach, Player_in_White).

animal:
- Any visible animal (dog, monkey, parrot, etc.).

object:
- Any physical object, tool, container, device, vehicle, furniture,
  signage, clothing item, or environmental object.

location:
- A visible physical area or environment
  (e.g., bench area, road, wall, ticket counter).

event (CORE ENTITY TYPE):
- A concrete, observable action or interaction.
- Includes:
  • holding,
  • placing,
  • moving,
  • sitting,
  • running,
  • facing,
  • pointing,
  • emerging (e.g., smoke from barrel),
  • state transitions (open/closed).

EVENT RULES:
- Events MUST be grounded strictly in the caption.
- Events MUST include a temporal_scope copied EXACTLY.
- Do NOT include timestamps inside entity_description.
- Each event must describe ONE coherent visible action pattern.

------------------------------------------------------------
Entity Fields
------------------------------------------------------------

For each entity, extract:

- entity_name:
  Canonical name (capitalized, consistent).

- entity_type:
  One of {entity_types}.

- entity_description:
  Factual description grounded strictly in caption.

- temporal_scope:
  REQUIRED only for event entities.
  Format: DAY# HH:MM:SS-HH:MM:SS.
  Copy exactly from caption.
  Leave EMPTY for non-event entities.

Format each entity as:
("entity"{tuple_delimiter}
 <entity_name>{tuple_delimiter}
 <entity_type>{tuple_delimiter}
 <entity_description>{tuple_delimiter}
 <temporal_scope or EMPTY>)

------------------------------------------------------------
B) Extract Relationships (EVENT-CENTRIC)
------------------------------------------------------------

General Rules:

- All actions must be mediated by event nodes.
- person → participates_in → Event
- animal → participates_in → Event
- Event → holds / places / moves / sits_on / runs_toward /
           faces / emerges_from / located_near / attached_to → Object
- Event → occurs_in → Location
- Event → follows / continues / interrupts → Event (if applicable)

Direct relationships between non-event entities are discouraged.
Prefer event-mediated structure.

------------------------------------------------------------
Relationship Fields
------------------------------------------------------------

For each relationship:

- source_entity
- target_entity
- relationship_type
- relationship_description
- relationship_strength (1-10)

relationship_strength guideline:
- 9–10 = central to scene understanding
- 6–8  = important contextual interaction
- 3–5  = background relevance

Format each relationship as:
("relationship"{tuple_delimiter}
 <source_entity>{tuple_delimiter}
 <target_entity>{tuple_delimiter}
 <relationship_type>{tuple_delimiter}
 <relationship_description>{tuple_delimiter}
 <relationship_strength>)

------------------------------------------------------------
Coverage Requirements
------------------------------------------------------------

You MUST capture:

- number of people visible (each as separate entity),
- who holds what,
- spatial relations (left/right/front/behind),
- orientation (facing left/right),
- attribute cues (color of clothing, object color),
- visible text as object entities,
- what emerges from objects (e.g., smoke from barrel),
- entering or leaving view.

Do NOT:

- infer intent,
- assume unseen actions,
- create abstract logical entities,
- invent events not in caption.

------------------------------------------------------------
C) Output
------------------------------------------------------------

Return a SINGLE list containing:

1) All extracted entities FIRST.
2) All extracted relationships AFTER.

Use {record_delimiter} as list delimiter.

------------------------------------------------------------
D) Completion
------------------------------------------------------------

When finished, output {completion_delimiter}.

######################
-Example-
######################

Entity_types: [person, animal, location, object, event]

Text:
DAY3 10:15:00-10:15:20:
Two men are sitting on a wooden bench. A woman stands to the right of the bench. One man is holding a green bag. The other man is looking to the left. Smoke is coming out of a barrel behind them.

Output:
("entity"{tuple_delimiter}"Man_1"{tuple_delimiter}"person"{tuple_delimiter}"A man sitting on a wooden bench."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Man_2"{tuple_delimiter}"person"{tuple_delimiter}"Another man sitting on the same bench."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Woman_1"{tuple_delimiter}"person"{tuple_delimiter}"A woman standing to the right of the bench."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Wooden Bench"{tuple_delimiter}"object"{tuple_delimiter}"A bench on which two men are sitting."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Green Bag"{tuple_delimiter}"object"{tuple_delimiter}"A green bag held by one of the men."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Barrel"{tuple_delimiter}"object"{tuple_delimiter}"A barrel positioned behind the bench."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Smoke"{tuple_delimiter}"object"{tuple_delimiter}"Smoke emerging from the barrel."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"E_SITTING_EVENT"{tuple_delimiter}"event"{tuple_delimiter}"Two men sit on a wooden bench while a woman stands nearby."{tuple_delimiter}"DAY3 10:15:00-10:15:20"){record_delimiter}
("entity"{tuple_delimiter}"E_HOLDING_BAG"{tuple_delimiter}"event"{tuple_delimiter}"One man holds a green bag while seated on the bench."{tuple_delimiter}"DAY3 10:15:00-10:15:20"){record_delimiter}
("entity"{tuple_delimiter}"E_SMOKE_EMERGENCE"{tuple_delimiter}"event"{tuple_delimiter}"Smoke emerges from the barrel behind the bench."{tuple_delimiter}"DAY3 10:15:00-10:15:20"){record_delimiter}

("relationship"{tuple_delimiter}"Man_1"{tuple_delimiter}"E_SITTING_EVENT"{tuple_delimiter}"participates_in"{tuple_delimiter}"Man_1 is seated on the bench."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Man_2"{tuple_delimiter}"E_SITTING_EVENT"{tuple_delimiter}"participates_in"{tuple_delimiter}"Man_2 is seated on the bench."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Woman_1"{tuple_delimiter}"E_SITTING_EVENT"{tuple_delimiter}"participates_in"{tuple_delimiter}"Woman_1 stands near the bench."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"E_HOLDING_BAG"{tuple_delimiter}"Green Bag"{tuple_delimiter}"holds"{tuple_delimiter}"The green bag is held by one of the seated men."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"E_SMOKE_EMERGENCE"{tuple_delimiter}"Smoke"{tuple_delimiter}"emerges_from"{tuple_delimiter}"Smoke comes out of the barrel."{tuple_delimiter}9){completion_delimiter}

######################
-Input-
Detailed Captions: {input_text}
Entity_types: {entity_types}
######################
Output:
"""

OVOBENCH_PROMPTS[
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

OVOBENCH_PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

OVOBENCH_PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

OVOBENCH_PROMPTS["proactive_service_prompt_"] = """
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

OVOBENCH_PROMPTS["proactive_service_prompt"] = """
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

OVOBENCH_PROMPTS["proactive_service_prompt_with_memory_simple"] = """
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

# ============================================================
# Forward Active Responding 任务（REC / SSR / CRR）相关 prompt
# ============================================================
# 这三类任务在每条样本的 test_info 中给出多个 realtime 时间点，
# 模型需要在每个时间点上根据"截至该时刻"的视觉证据给出一次回答：
#   - REC: 计数已完成的重复动作次数
#   - SSR: 判断当前是否正在执行某个 step
#   - CRR: 判断"最新画面"是否提供了足够回答 question 的线索
#
# 建图阶段我们仍然复用 simple_second_caption_system_prompt 作为
# 视觉证据型 caption 的总框架，只把 {questions} attention guidance
# 替换成下面三个 focus 文本。

OVOBENCH_PROMPTS["forward_rec_focus"] = """
This video may contain a repetitive action performed by one or
more people. The target action is:

ACTION: "{activity}"

For EACH frame you describe, you MUST:
- Explicitly note whether the target action is currently in progress.
- Identify which person is performing it (e.g., Man_1, Woman_2),
  using consistent identifiers across frames.
- Mark the START moment of one motion (when the action visibly begins).
- Mark the END moment of one motion (when the action visibly completes,
  i.e. one COMPLETE motion has finished).
- If a person merely holds a similar pose without performing the
  action, note that as "no active motion".
- If multiple people are performing the action concurrently, record
  each one separately so the count can be reconstructed later.

You MUST NOT count or summarize the number of motions; only record
the per-frame visible evidence. Counting will be done downstream.
"""

OVOBENCH_PROMPTS["forward_ssr_focus"] = """
This is a tutorial video. The full procedure consists of the
following ordered steps (use the numeric labels below to
disambiguate which step is being executed):

STEPS:
{numbered_steps}

For EACH frame you describe, you MUST:
- Identify which numbered step (if any) the person in view is
  currently performing, using the exact label "Step <N>" plus the
  step text.
- Provide concrete visual evidence (tools used, body parts, target
  object, action verb) supporting that judgement.
- If the person is between steps, idle, or performing something not
  listed, explicitly say "no listed step in progress".
- If a step is clearly being completed at this frame, mark it with
  "Step <N> completed".
- Do NOT predict future steps; only describe what is visible.

You MUST NOT answer whether a particular step is being performed
overall; only record the per-frame evidence.
"""

OVOBENCH_PROMPTS["forward_crr_focus"] = """
At the end of this video, a user will ask the following question.
Until then, your job is to record visual evidence that may later be
used to decide whether the question can be answered.

QUESTION: "{question}"

For EACH frame you describe, you MUST:
- Explicitly record any object, person, action, location, text or
  state that appears directly relevant to the question.
- Note whether the relevant cue is currently visible, partially
  visible, or absent.
- If a referenced entity (mentioned in the question) enters or leaves
  view, mark the transition.
- Be especially thorough about the LATEST frames within each window:
  whether the cue needed to answer the question is visible right now.

You MUST NOT answer the question. You MUST NOT speculate about
future frames. Only record observable evidence.
"""

# ----- 推理阶段 (per realtime point) -----

OVOBENCH_PROMPTS["forward_rec_inference_prompt"] = """
You are a video question-answering assistant for the OVO-Bench
Forward Active Responding (REC) task.

You are given multi-scale captions describing what was visible in
the video UP TO the current moment ({current_time}).
You must count how many COMPLETE motions of a specific repetitive
action have occurred in the video up to (and including) this moment.

------------------------------------------------------------
TARGET ACTION
------------------------------------------------------------
{activity}

------------------------------------------------------------
COUNTING RULES (STRICT)
------------------------------------------------------------
- One COMPLETE motion = the action visibly starts AND visibly ends.
- A motion that has only started but not yet finished by
  {current_time} MUST NOT be counted.
- Different people performing the action concurrently are counted
  independently. (E.g., 2 people each completing 1 motion = 2.)
- Do NOT count a person merely holding a similar pose.
- Use ONLY information present in the captions. Do NOT speculate.
- Online setting: information after {current_time} is not available.

------------------------------------------------------------
INPUTS
------------------------------------------------------------

CURRENT_TIME: {current_time}

SECOND_CAPTIONS (~10s windows, all up to {current_time}):
{second_captions_text}

GLOBAL_CAPTIONS (~1min summaries up to {current_time}):
{global_captions_text}

------------------------------------------------------------
REASONING PROCEDURE
------------------------------------------------------------
1. Walk through SECOND_CAPTIONS in chronological order.
2. List EVERY candidate "complete motion" event you can find,
   recording (performer, start_time, end_time).
3. Drop any candidate whose end is after {current_time} or whose
   end is not visible.
4. Drop duplicates (same performer, overlapping time spans).
5. Count the remaining events.

------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------------------------------------------
{{
  "count": <non-negative integer>,
  "evidence": [
     {{"performer": "<id>", "start": "<HH:MM:SS or sec>",
       "end": "<HH:MM:SS or sec>"}},
     ...
  ],
  "reasoning": "<one short sentence>"
}}

Output ONLY the JSON object. No markdown.
"""

OVOBENCH_PROMPTS["forward_ssr_inference_prompt"] = """
You are a video question-answering assistant for the OVO-Bench
Forward Active Responding (SSR) task.

You are given multi-scale captions describing what was visible in
this tutorial video UP TO the current moment ({current_time}).
You must decide whether the person in the video is CURRENTLY (at
{current_time}) performing the specific step listed below.

------------------------------------------------------------
ALL STEPS IN THE TUTORIAL (for context / disambiguation)
------------------------------------------------------------
{numbered_steps}

------------------------------------------------------------
QUERIED STEP (the only one to judge)
------------------------------------------------------------
Step {target_step_index}: {target_step_text}

------------------------------------------------------------
DECISION RULES (STRICT)
------------------------------------------------------------
- Answer "Yes" ONLY IF the RECENT_CAPTIONS provide direct visual
  evidence that the person is performing the queried step at
  {current_time} (i.e. the action is currently in progress, not
  finished, not yet started).
- Answer "No" if:
  * the queried step has clearly already been completed earlier and
    the person has moved on,
  * the queried step has not started yet,
  * the person is performing a different listed step,
  * the person is idle / between steps,
  * RECENT_CAPTIONS do not provide direct evidence of the step.
- GLOBAL_CAPTIONS may be used to confirm context (e.g., earlier
  steps were already completed) but the trigger for "Yes" MUST come
  from RECENT_CAPTIONS.
- Online setting: information after {current_time} is not available.

------------------------------------------------------------
INPUTS
------------------------------------------------------------

CURRENT_TIME: {current_time}

RECENT_CAPTIONS (last ~10s of second-level captions):
{recent_captions_text}

GLOBAL_CAPTIONS (~1min summaries up to {current_time}):
{global_captions_text}

------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------------------------------------------
{{
  "answer": "Yes" | "No",
  "reasoning": "<one short sentence grounded in captions>"
}}

Output ONLY the JSON object. No markdown.
"""

OVOBENCH_PROMPTS["forward_crr_inference_prompt"] = """
You are a video question-answering assistant for the OVO-Bench
Forward Active Responding (CRR — Clue Reveal Responding) task.

You are given multi-scale captions describing what was visible in
the video UP TO the current moment ({current_time}). A user is
asking the following question, which is intended to be answered
based on what is visible NEAR THE END of the video so far (i.e. the
latest frames at {current_time}).

You MUST decide whether the existing visual evidence — especially
the LATEST frames at {current_time} — already contains enough cues
to answer the question. You MUST NOT actually answer the question.

------------------------------------------------------------
QUESTION
------------------------------------------------------------
{question}

------------------------------------------------------------
DECISION RULES (STRICT)
------------------------------------------------------------
- Answer "Yes" ONLY IF the RECENT_CAPTIONS (or, when complementary,
  RECENT_CAPTIONS + relevant past evidence) make the answer to the
  question visually determinable right now.
- Answer "No" if:
  * the entities/actions referenced by the question are not yet
    visible,
  * the cue has already passed and is no longer present in the
    latest frames,
  * the latest frames do not provide the determining evidence.
- The judgement MUST be anchored on the LATEST frames; mere
  historical mentions in GLOBAL_CAPTIONS are insufficient unless
  the latest frames also show a relevant cue.
- Online setting: information after {current_time} is not available.

------------------------------------------------------------
INPUTS
------------------------------------------------------------

CURRENT_TIME: {current_time}

RECENT_CAPTIONS (last ~10s of second-level captions):
{recent_captions_text}

GLOBAL_CAPTIONS (~1min summaries up to {current_time}):
{global_captions_text}

------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------------------------------------------
{{
  "answer": "Yes" | "No",
  "reasoning": "<one short sentence grounded in captions>"
}}

Output ONLY the JSON object. No markdown.
"""

OVOBENCH_PROMPTS["DEFAULT_ENTITY_TYPES"] = ["person", "location", "object", "event", "animal"]
OVOBENCH_PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
OVOBENCH_PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
OVOBENCH_PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
OVOBENCH_PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
OVOBENCH_PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
OVOBENCH_PROMPTS["default_text_separator"] = [
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


OVOBENCH_PROMPTS["caption_reconstruction"] = """
You are a visual evidence re-captioning assistant
for the OVO-Bench benchmark.

------------------------------------------------------------
Goal
------------------------------------------------------------

Given retrieval keywords and a short video segment,
rewrite ONE refined caption that:

1) Explicitly highlights visual evidence most relevant to the retrieval keywords, AND
2) Supplements other clearly visible objects, actions, or states
   that help clarify the scene configuration.

The output is NOT an answer.
It is a refined visual evidence description.

------------------------------------------------------------
Inputs
------------------------------------------------------------

• Retrieval keywords derived from the user question.
• A short video segment (~5–10 seconds),
  sampled frames in temporal order.
• An ORIGINAL fine-grained caption for the same segment
  (reference only).

------------------------------------------------------------
Perspective Handling (CRITICAL)
------------------------------------------------------------

The video may be:

• First-person (egocentric), OR  
• Third-person (external view).

You MUST:

• Use "I" ONLY if the frames clearly indicate
  egocentric first-person perspective.

• Otherwise, use neutral third-person descriptions such as:
  - "a person", "the man", "the woman", "someone",
  - or specific visible entities when identifiable.

Do NOT assume egocentric perspective.

------------------------------------------------------------
Output
------------------------------------------------------------

• Output EXACTLY ONE caption.
• No JSON or special formatting.
• Grounded strictly in what is visible in the frames.

------------------------------------------------------------
How to Use the Retrieval Keywords (MANDATORY)
------------------------------------------------------------

You MUST use the retrieval keywords to guide attention.

For each visible keyword-related element:

• explicitly name the object,
• describe its visible state,
• describe its spatial relation to other entities,
• describe any visible motion or state change.

If a keyword object appears,
it MUST be named using a concrete noun.

------------------------------------------------------------
Keyword-Type-Specific Reinforcement
------------------------------------------------------------

1) If keywords involve state change or spatial relation
   (e.g., closer/farther, left/right, enter/leave view):

   → You MUST describe:
     - which entity moves,
     - visible directional change,
     - relative position between entities,
     - continuity if no change occurs.

2) If keywords involve actions:

   → You MUST:
     - clearly describe the action,
     - describe progression across frames,
     - describe interaction with specific objects,
     - indicate continuation, pause, or change.

3) If keywords involve text or numbers:

   → You MUST:
     - transcribe ALL clearly readable text exactly as seen,
     - include numbers and symbols,
     - describe where the text appears.

4) If keywords involve object category or attributes:

   → You MUST:
     - specify object type,
     - describe visible color, material, size,
     - describe precise spatial placement,
     - indicate whether stationary or moving.

------------------------------------------------------------
Supplementary Evidence Requirement
------------------------------------------------------------

In addition to keyword-related elements,
you SHOULD include other clearly visible objects
that clarify:

• spatial layout,
• entity relationships,
• background structure.

Do NOT hallucinate.
Do NOT introduce unseen elements.

------------------------------------------------------------
Original Caption Usage
------------------------------------------------------------

The original caption is reference only.

Frames override the original caption.

Do NOT copy it verbatim.
Do NOT rely on it if it omits visible evidence.

------------------------------------------------------------
Caption Content Requirements
------------------------------------------------------------

Describe only what is visually observable:

• visible actions and interactions,
• object states (open/closed, held/placed, visible/blocked),
• spatial relations between entities,
• orientation (facing left/right/front/back),
• visible motion or lack of motion.

If a keyword-related change occurs,
you MUST describe it explicitly.

------------------------------------------------------------
Constraints
------------------------------------------------------------

• Do NOT infer intent, purpose, or correctness.
• Do NOT answer the question.
• Do NOT speculate about unseen future events.
• Keep the caption factual and visually grounded.
• One continuous caption (1–5 sentences).

------------------------------------------------------------
Important Notes
------------------------------------------------------------

• Every visible keyword-related element MUST be described.
• Rich spatial grounding improves retrieval quality.
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

OVOBENCH_PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """
------------------------------------------------------------
- Goal (OVO-Bench Entity & Event Retrieval Query) -
------------------------------------------------------------

Given a question from the OVO-Bench benchmark,
rewrite it as EXACTLY ONE concise declarative sentence
that can be used to retrieve relevant entities and events
from an EVENT-CENTRIC visual knowledge graph.

The rewritten sentence should describe
the observable action, object, state change,
spatial relation, or interaction
that should appear in memory at or before the query timestamp.

The output is NOT an answer.
It is a retrieval-oriented description of visual evidence.

------------------------------------------------------------
Core Principle (Entity- and Event-Oriented)
------------------------------------------------------------

You are translating a question into a
VISUAL-EVIDENCE QUERY aligned with
how entities and events are stored in memory.

The query should emphasize:

- concrete objects (explicitly named),
- observable actions performed by a visible person,
- object state changes,
- spatial relations between entities,
- visible motion or direction,
- repetition if implied.

All described content MUST be visually observable.

------------------------------------------------------------
Perspective Handling (CRITICAL)
------------------------------------------------------------

The video may be:

• First-person (egocentric), OR  
• Third-person (external view).

You MUST:

• Use neutral descriptions such as:
  - "a person", "the man", "the woman", "someone",
  when the viewpoint is third-person.
• Use "I" ONLY if the original question clearly refers to the camera wearer.

Do NOT assume egocentric perspective unless explicitly stated.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Do NOT ask a question.
- Do NOT include explanations, reasoning, or advice.
- Do NOT include abstract goals or inferred intent.
- Do NOT assume future events beyond the query moment.

------------------------------------------------------------
What to Express (KEEP CONCRETE)
------------------------------------------------------------

The rewritten sentence SHOULD include one or more of:

- explicit object names (e.g., knife, exit sign, chair, laptop),
- observable actions (e.g., walking, cutting, placing, sitting),
- visible state or position changes,
- spatial relations (e.g., near the door, behind the table),
- orientation (e.g., facing left, facing the camera),
- repeated visible events if implied.

Prefer:
- object-centric phrasing,
- entity- and event-aligned descriptions,
- literal visual language.

Avoid:
- high-level summaries,
- inferred purpose,
- abstract task labels.

------------------------------------------------------------
Temporal Awareness (Online Setting)
------------------------------------------------------------

If the question implies temporal reasoning (e.g., before, after, when),
describe only observable past or current events.

Do NOT:
- reference future information,
- invent durations or time spans.

------------------------------------------------------------
Examples (OVO-Aligned)
------------------------------------------------------------

Question:
When does the red exit change its position relative to the person?
Output:
Events where the person moves and a red exit sign becomes closer to or farther from them.

Question:
Is the kettle visible to the other person after it is turned off?
Output:
Events where a person turns off a kettle and the kettle remains visible or becomes blocked from another person’s view.

Question:
How many people are sitting at the table?
Output:
Events where multiple people are visible sitting around a table.

Question:
What object does the woman pick up?
Output:
Events where a woman reaches for and lifts a visible object.

Question:
What happens before the man leaves the room?
Output:
Events showing the man’s actions immediately before he exits the room.

------------------------------------------------------------
Output
------------------------------------------------------------

Output exactly ONE declarative sentence suitable for
entity and event retrieval.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------
Question: {input_text}
------------------------------------------------------------
Output:
"""

OVOBENCH_PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """
------------------------------------------------------------
- Goal (OVO-Bench → Visual Embedding Retrieval Query) -
------------------------------------------------------------

Given a multiple-choice question from OVO-Bench,
rewrite it as EXACTLY ONE concise English declarative sentence
to retrieve visually relevant video segments using visual embeddings.

The rewritten sentence should describe
WHAT VISUAL CONTENT is likely to appear in segments
that help distinguish between the answer options.

The output is NOT an answer.
It is a VISUAL RETRIEVAL QUERY only.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Do NOT ask a question.
- Do NOT include explanations or reasoning.
- Do NOT state conclusions.

------------------------------------------------------------
Perspective Handling (CRITICAL)
------------------------------------------------------------

The video may be:

• First-person (egocentric), OR  
• Third-person (external view).

You MUST:

• Use neutral phrasing such as:
  - "a person", "the man", "the woman", "someone",
  unless the question explicitly refers to the camera wearer.
• Use "I" ONLY when clearly specified.

Do NOT assume egocentric perspective.

------------------------------------------------------------
Visual Grounding Rules (MANDATORY)
------------------------------------------------------------

The query MUST describe ONLY visually observable content, including:

- visible actions (walking, sitting, picking up, placing, turning, facing),
- concrete objects (chair, laptop, knife, exit sign, bucket),
- spatial relations (near the door, behind the table, to the left of a person),
- orientation (facing left, facing the camera),
- visible attributes (color, clothing, size),
- number of entities if counting is implied,
- visible motion or position change.

You MUST NOT:

- assume a change happened unless visually described,
- infer intention or purpose,
- describe unseen future outcomes,
- rely on world knowledge.

------------------------------------------------------------
Multiple-Choice Handling (IMPORTANT)
------------------------------------------------------------

If the question is abstract or global:

• Convert each option into a concrete, visually distinguishable alternative.
• Express them as observable visual possibilities in the query.
• Use visual OR conditions when needed.

Examples:
- different object types,
- different actions,
- different locations,
- different spatial arrangements.

------------------------------------------------------------
Temporal Awareness (Online Setting)
------------------------------------------------------------

OVO-Bench follows online video understanding.

• Describe only observable past or current events.
• Do NOT reference future frames.
• Avoid explicit temporal words unless directly implied.

------------------------------------------------------------
Examples (OVO-Aligned)
------------------------------------------------------------

Question:
When does the red exit change its position relative to the person?
Output:
A moment where a person moves and a red exit sign shifts position in the frame relative to them.

Question:
How many people are sitting at the table?
Output:
Moments where multiple people are visibly seated around a table.

Question:
Is the kettle visible after it is turned off?
Output:
A moment where a person turns off a kettle and the kettle remains visible or becomes partially obscured.

Question:
What object does the woman pick up?
Output:
A moment where a woman reaches toward and lifts a visible object from a surface.

Question:
What happens before the man leaves the room?
Output:
Moments showing the man’s actions immediately prior to exiting a room.

------------------------------------------------------------
Output
------------------------------------------------------------

Output exactly ONE declarative sentence suitable for
visual embedding retrieval.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------
Question: {input_text}
------------------------------------------------------------
Output:
"""

OVOBENCH_PROMPTS[
    "keywords_extraction"
] = """
------------------------------------------------------------
- Goal (OVO-Bench → Caption Reconstruction Keywords) -
------------------------------------------------------------

Given ONE question from OVO-Bench retrieval,
extract a compact set of VISUAL keywords for caption reconstruction.

These keywords will be used to re-check frames
and refine a visual caption.

The keywords must prioritize:

(1) the queried OBJECTS (with attributes such as color/type if present),
(2) the queried ACTIONS / INTERACTIONS,
(3) the queried SPATIAL RELATIONS between entities,
(4) the queried STATE / ORIENTATION changes,
(5) any VISUALLY OBSERVABLE TEXT/NUMBERS if relevant.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output: comma-separated keywords in English; NO extra text.
- Prefer 3–7 keywords total (minimal but sufficient).
- Use short noun phrases / verb phrases (1–4 words each).
- REMOVE function words and meta-words such as:
  "when", "does", "how many", "can you", "video", "moment", "segment".
- Keep only VISUAL-EVIDENCE-oriented terms:
  objects, attributes, visible actions, spatial relations, orientation, text.

- If the question involves:
  • counting → include object + "number of"
  • orientation → include "facing", "direction"
  • spatial relation → include "left of", "behind", "near", etc.
  • state change → include "picked up", "put down", "open", "closed", "on", "off"
  • visibility → include "visible", "in view", "blocked"
  • text → include "label", "text", "number"

- Do NOT add abstract intent or purpose.
- Do NOT include time hints unless visually grounded.

------------------------------------------------------------
Examples (OVO-Bench Aligned)
------------------------------------------------------------

Q: How many people are sitting at the table?
Output:
people, sitting, table, number of, chairs

Q: Is the kettle visible after it is turned off?
Output:
kettle, turned off, visible, person, countertop

Q: What object does the woman pick up?
Output:
woman, picked up, object, table, hands

Q: Which direction is the man facing?
Output:
man, facing direction, body orientation, head direction

Q: What is written on the blue sign?
Output:
blue sign, label, text, printed words

Q: What happens before the man leaves the room?
Output:
man, walking, door, exit, room

Q: Where is the backpack placed?
Output:
backpack, placed, floor, beside chair, location

-------------------------
Real Data
-------------------------
Question: {input_text}
Output:
"""
