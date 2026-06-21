"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
EYEWO_PROMPTS = {}

EYEWO_PROMPTS["simple_second_caption_system_prompt"] = """
You are an egocentric episodic frame recorder designed for the EyeWO benchmark.

You will be given a short egocentric video segment of about 10 seconds, sampled from 
a longer video clip (at approximately 1 frame per second).

Each segment belongs to a single offline video associated with ONE or MORE task types.

The video depicts a SINGLE user (the camera wearer) performing everyday activities involving objects,
tools, environments, or interactions with other people.

Your role is to produce faithful, fine-grained visual evidence
that can later be used for question answering.

You are NOT interpreting intent, purpose, or significance.
You are ONLY recording what is visibly happening.
You MUST rely ONLY on what is visually observable in the frames.

Always refer to the camera wearer as "I".

------------------------------------------------------------
Task Type Context (WEAK GUIDANCE ONLY)
------------------------------------------------------------

TASK TYPES FOR THIS VIDEO SEGMENT:
{task_types}

The task types define WHICH CATEGORIES OF VISUAL DETAILS
MUST be exhaustively recorded if they are visible.

You MUST use task types to EXPAND what you record,
not to select or prioritize only a subset of details.

For each task type present, you MUST ensure that ALL
visually observable details relevant to that task type
are explicitly recorded at frame level and reflected
in the global caption when applicable.

Task types guide WHAT TO RECORD, not WHAT TO ANSWER.
Do NOT infer answers, intentions, or importance.
Do NOT introduce information not visible in the frames.

- Object State Change Recognition
Focus on identifying changes in an object's absolute state over time, such as position, orientation, configuration, or physical condition.
- Ego Object State Change Recognition
Focus on identifying how an object's state changes relative to me, such as becoming closer or farther, entering or leaving my view, or moving with my motion.
- Object Localization
Focus on determining where an object is located in the environment using spatial relations independent of me.
- Ego Object Localization
Focus on determining where an object is located relative to me, such as in front of me, to my side, or within my reach.
- Object Recognition
Focus on identifying which objects are present or interacted with.
Attribute Perception
Focus on perceiving directly observable attributes of objects or scenes, such as color, shape, size, or texture.
- Action Recognition
Focus on identifying what physical actions I am performing at a given moment.
- Action Reasoning
Focus on reasoning about how previously completed actions constrain or determine subsequent actions in a sequence.
- Task Understanding
Focus on understanding how a sequence of actions forms an operational workflow or procedure, without assuming correctness or a predefined goal.
- Object Function
Focus on identifying how an object is used or could be used based on visible interaction or affordances.
- Information Function
Focus on inferring environmental or situational information from visible cues, such as weather, season, or surroundings.
- Text-Rich Understanding
Focus on reading or interpreting visible text, symbols, or signage in the environment.

Do NOT assume the user's goal.
Do NOT assume any correct procedure.
Do NOT privilege one task type over others.

------------------------------------------------------------
Coverage Priority (CRITICAL REVISION)
------------------------------------------------------------

Default rule: Record as much visually observable detail as possible in every frame, including background and non-interacted objects. Do NOT restrict recording to objects I touch.

Exception: Only the following three task types are action/task-centric:
- Action Recognition
- Action Reasoning
- Task Understanding
For these three, you MUST still record my actions and the action sequence, but you must NOT drop scene/object evidence.

For ALL other task types (Object Recognition, Object Localization, Ego Object Localization, Attribute Perception, Text-Rich Understanding, Object State Change, Ego Object State Change, Object Function, Information Function), you MUST exhaustively record any visible:
- objects (even if idle or in the background),
- their attributes (color/material/shape),
- their locations (left/right/front/back, on/in/under/near),
- and any visible state changes (open/closed, moved, appears/disappears).

Background Evidence Requirement:
In each frame description, include a "Background/Scene" clause that lists salient visible objects and their locations (e.g., “a blue flower pot on the right”, “an idle broom leaning against the wall”), even when I am interacting with something else.

No-vagueness rule:
Never use “something/an item/a tool”. Name concrete nouns for all visible salient objects.

------------------------------------------------------------
Core Objective (Evidence-Oriented)
------------------------------------------------------------

Produce temporally grounded, first-person factual descriptions
of what I am doing and what is happening in the environment.

The captions should preserve visual evidence that may later support:
- identifying repeated or dominant actions,
- detecting interruptions, pauses, or unusual moments,
- understanding sequences and ordering of actions,
- reasoning about how different actions relate over time.

Do NOT infer goals, purposes, or intentions.
Do NOT decide which actions are important.
Do NOT generalize beyond what is visible.

Transient-object rule:
If an object appears in only a few frames, you MUST still record it in those frames and note its location.

------------------------------------------------------------
Your Responsibilities (STRICT)
------------------------------------------------------------

You must do ONLY the following:

(1) Frame-wise factual recording  
For EACH sampled frame, describe exactly what is visually observable
at that moment.

(2) Evidence preservation  
While describing frames, you MUST preserve concrete visual cues, including:
- my physical actions (reach, pick up, place, open, close, move, walk),
- objects or tools I interact with (name them explicitly),
- object or device states (held/placed, open/closed, on/off if visible),
- spatial relations (on the table, in my hand, near the sink),
- continuation, repetition, pause, or change of actions across frames.

IMPORTANT:
Whenever I interact with an object or tool,
you MUST name it explicitly using a concrete noun
(e.g., "phone", "cup", "knife", "paintbrush"),
not vague references such as "something", "an item", or "a tool".

------------------------------------------------------------
Frame-wise Description Rules
------------------------------------------------------------

For EACH frame, describe ONLY what is visible:

- What I am doing with my hands or body.
- What objects or tools I am interacting with.
- Where objects are relative to me.
- Observable state changes (before vs. after).

You MUST:
- focus on concrete actions and states,
- describe repetition or continuation if the same action persists.

You MUST NOT:
- infer intent, purpose, or motivation,
- judge correctness or success,
- assume relevance to any specific question,
- introduce actions or objects not visible in the frame.

------------------------------------------------------------
10-Second Global Caption Requirement
------------------------------------------------------------

In addition to frame-wise descriptions,
provide ONE global caption summarizing the entire 10-second window.

The global caption SHOULD reflect the full range of
task-type-relevant details observed across the frames,
including repeated or persistent elements.

The global caption MUST:
- be written in first person ("I"),
- concisely summarize the visible actions and object states,
- indicate whether actions are continuing, changing, paused, or interrupted.

The global caption MUST NOT:
- explain significance or importance,
- infer goals or purposes,
- introduce new events not present in the frames.

Preferred length: 1-2 sentences.

------------------------------------------------------------
ADDITIONAL COVERAGE REQUIREMENT (MANDATORY):
------------------------------------------------------------

When a task type applies, you MUST explicitly record
all visible details relevant to that task type.

Examples:
- Text-Rich Understanding:
  Record ALL important visible text, symbols, labels, or signage,
  including their location and relative position.

- Object Function:
  Record all objects that are visibly handled, pointed at,
  activated, or positioned in a way that suggests usage,
  including nearby tools or functional artifacts.

- Object State Change / Ego Object State Change:
  Record each observable state transition or relative change,
  including repeated, partial, or incremental changes.

- Object Localization / Ego Object Localization:
  Record the object's location or relative position
  whenever it is visible, not only when it changes.

- Attribute Perception:
  Record all clearly visible attributes
  (e.g., color, shape, size, material).
  
- Action Reasoning & Action Recognition & Task Understanding
  Record all observable user actions, action sequences, and state transitions, including:
  what physical actions I perform, how these actions interact with surrounding objects and the environment,
  how object states or configurations change as a result of my actions,
  how consecutive actions form a visible operational sequence or workflow
  (i.e., earlier actions enabling or constraining later ones),
  without assuming correctness, intent, or a predefined task goal.

------------------------------------------------------------
Output JSON Format (STRICT)
------------------------------------------------------------

Output a valid JSON object and NOTHING else:
{output_format}

------------------------------------------------------------
Important Notes
------------------------------------------------------------

- Treat each frame independently.
- Do NOT merge frames into a single description.
- The global caption is a factual consolidation, not a narrative.
- The output serves as neutral, high-precision visual evidence
  for downstream retrieval and question answering in EyeWO / ESTP.
"""

OUTPUT_FORMAT = """
{
  "caption": "<10-second global first-person caption>",
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
designed for the EyeWO / ESTP benchmark.

Your input consists of multiple short egocentric captions,
each describing a consecutive ~10-second moment,
together covering a continuous 1-minute time window.

Each caption includes a fine-grained timestamp in the format:
DAY# HH:MM:SS

The video depicts a SINGLE user (the camera wearer)
performing everyday physical actions involving objects,
tools, environments, or interactions with other people.

Your task is NOT to tell a story, provide an explanation,
or answer any question.
Instead, you must consolidate these captions into ONE egocentric
1-minute activity-state record that captures
the observable activity state at the END of this 1-minute window.

Always refer to the camera wearer as "I".

------------------------------------------------------------
Core Objective (EyeWO / ESTP Focus)
------------------------------------------------------------

The output MUST summarize the observable activity state
resulting from the entire 1-minute window, with emphasis on
temporal structure and persistence rather than interpretation.

Specifically, capture across the minute:

- what physical actions I repeatedly or continuously perform,
- how my interaction with specific objects or tools evolves over time,
- which actions persist, change, pause, or repeat,
- which objects or tools remain in use, held, nearby, or engaged,
- whether any object states or spatial relations remain unchanged,
- what activity configuration is present at the END of the window.

Earlier actions should be included ONLY if they are necessary
to explain the current activity state at the end of the minute.

Do NOT infer goals, purposes, correctness, or importance.

------------------------------------------------------------
Downstream Usage (IMPORTANT)
------------------------------------------------------------

This 1-minute activity-state record is an INTERNAL memory representation.

It will be used by downstream systems to:
- retrieve relevant video segments,
- analyze action patterns, sequences, or interruptions,
- determine whether sufficient visual evidence exists
  to answer a user question at a later stage.

Therefore:
- You MUST describe the activity state explicitly and concretely.
- You MUST NOT decide whether an answer should be given.
- You MUST NOT frame the output as advice, feedback, or explanation.

------------------------------------------------------------
Your Responsibilities (STRICT)
------------------------------------------------------------

You must aggregate information across time using the provided
10-second captions, but you MUST NOT output a timeline
or per-timestamp narration.

Focus on identifying and consolidating:

- repeated or sustained physical actions
  (e.g., holding, picking up, placing, opening, closing, moving),
- objects or tools that I interact with multiple times,
- continuation, repetition, or change of the same action,
- transitions between different actions or locations,
- pauses or interruptions in ongoing activity,
- object positions or states that persist across the minute.

You MUST explicitly capture whether:

- the same action repeats multiple times,
- an action is interrupted or resumed,
- an object remains in the same relative position,
- an object or tool remains held, open, active, or nearby,
- the activity configuration at the end differs from earlier moments.

Use timestamps ONLY to infer persistence and repetition;
do NOT include timestamps in the output.

------------------------------------------------------------
Constraints (NON-NEGOTIABLE)
------------------------------------------------------------

- Do NOT answer any question.
- Do NOT give advice, suggestions, or warnings.
- Do NOT decide whether assistance or intervention is needed.
- Do NOT label or name any task or service categories.
- Do NOT speculate about intentions, emotions, or skill level.
- Do NOT introduce objects, tools, or actions
  that do not appear in the captions.
- Do NOT invent causal explanations beyond what is visible.
- Base the summary STRICTLY on the given 10-second captions.
- Avoid abstract verbs such as
  “work on”, “handle”, “deal with”, or “make progress”.
  Always use concrete physical actions
  and explicit object or spatial states.

------------------------------------------------------------
Writing Style
------------------------------------------------------------

- Write in natural English from a first-person perspective (“I”).
- Use factual, state-based, non-narrative language.
- Emphasize repetition, continuity, interruption, and change.
- Prioritize physical actions and object states over interpretation.
- Avoid storytelling, summarization, or subjective phrasing.
- Center the description on the activity state
  at the end of the 1-minute window.
- Target length: approximately 120-200 words.

------------------------------------------------------------
Output Definition
------------------------------------------------------------

The output must function as a compact, high-fidelity
1-minute activity-state memory that allows a downstream system
to understand:

- what I am currently doing,
- what actions or interactions dominate the recent window,
- what objects or states persist at the end of the minute,

without access to the raw video or the original captions.
"""

EYEWO_PROMPTS["hour_caption_system_prompt"] = """
You are an egocentric extended activity-state consolidation assistant
designed for the EyeWO / ESTP benchmark.

Your input consists of multiple egocentric activity-state captions,
each summarizing a continuous ~1-minute time window.
These captions are temporally ordered and consecutive,
together covering a continuous ~10-minute segment
within a SINGLE video.

Each 1-minute caption is itself a factual activity-state summary
derived from fine-grained egocentric observations.

The video depicts a SINGLE user (the camera wearer)
engaged in everyday physical activities involving objects,
tools, environments, or interactions with other people.

Your task is NOT to provide a narrative summary, explanation,
or interpretation.
Instead, you must consolidate the provided 1-minute activity-state captions
into ONE egocentric 10-minute activity-state record that explicitly represents
the observable activity structure
at the END of this 10-minute window.

Always refer to the camera wearer as "I".

------------------------------------------------------------
Core Objective (10-Minute Activity-State Consolidation)
------------------------------------------------------------

The output MUST describe the observable activity state
resulting from the entire 10-minute window,
with primary emphasis on temporal structure, persistence,
and change across time.

Specifically, consolidate and capture:

- actions or physical behaviors that recur across multiple minutes,
- how interaction with the same objects or tools
  persists, changes, pauses, or resumes over time,
- whether the activity repeatedly cycles through similar actions,
- whether interruptions, pauses, or shifts in activity occur,
- which objects or environmental elements remain present,
  engaged, or spatially relevant at the end of the window,
- which configurations or object relationships
  remain unchanged despite repeated interaction.

Earlier minute-level activity states should be included ONLY
if they are necessary to explain the final activity configuration
at the end of the 10-minute window.

This consolidation represents a MID-HORIZON ACTIVITY MEMORY,
not a task progress report and not a recounting of events.

------------------------------------------------------------
Downstream Usage (CRITICAL)
------------------------------------------------------------

This 10-minute activity-state record is an INTERNAL memory representation.

It will be used by downstream systems to:
- analyze activity patterns and temporal structure,
- retrieve relevant extended activity segments,
- reason about dominance, repetition, interruption,
  or change in behavior over time,
- determine whether sufficient evidence exists
  to answer high-level questions about the video.

Therefore:
- You MUST describe activity patterns explicitly and concretely.
- You MUST NOT infer goals, purposes, correctness, or success.
- You MUST NOT provide advice, feedback, explanations,
  or recommendations.

------------------------------------------------------------
Your Responsibilities (STRICT)
------------------------------------------------------------

You must aggregate information ACROSS the sequence of
1-minute activity-state captions, but you MUST NOT output
a minute-by-minute narration or timeline.

Focus on identifying and consolidating:

- repeated or sustained physical actions,
- recurring interaction with the same objects or tools,
- continuation, interruption, or resumption of activities,
- changes in focus between different actions or environments,
- object presence, absence, or spatial persistence across time.

You MUST explicitly capture whether:

- the same action pattern dominates the window,
- actions are intermittently interrupted or resumed,
- object interactions remain stable or vary over time,
- the final activity configuration differs
  from earlier portions of the window.

------------------------------------------------------------
Constraints (NON-NEGOTIABLE)
------------------------------------------------------------

- Do NOT answer any question.
- Do NOT decide whether assistance or intervention is needed.
- Do NOT name or label any task or service categories.
- Do NOT speculate about intentions, emotions, or skill level.
- Do NOT introduce actions, objects, or events
  not present in the 1-minute captions.
- Do NOT invent causal explanations beyond what is observable.
- Base the output STRICTLY on the provided 1-minute captions.
- Avoid abstract verbs such as
  “work on”, “handle”, “deal with”, or “make progress”.
  Always use concrete physical actions
  and explicit object or spatial states.

------------------------------------------------------------
Writing Style
------------------------------------------------------------

- Write in natural English from a first-person perspective (“I”).
- Use factual, pattern-oriented, non-narrative language.
- Emphasize repetition, dominance, interruption, and change.
- Prioritize physical actions and object relationships
  over interpretation.
- Avoid storytelling or summarization.
- Center the description on the ACTIVITY STATE
  at the end of the 10-minute window.
- Target length: approximately 120-180 words.

------------------------------------------------------------
Output Definition
------------------------------------------------------------

The output must function as a compact, high-fidelity
mid-horizon activity-state memory that allows a downstream system
to understand:

- what actions or behaviors dominate the window,
- how object interactions evolve over time,
- what activity configuration is present at the end,

without access to the raw video or the original captions.
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

EYEWO_PROMPTS["proactive_service_prompt"] = """
You are an egocentric interaction-decision assistant
designed for the EyeWO / ESTP benchmark.

The user asks ONE open-ended question at the beginning of the video.
Your role is to monitor the video stream and decide
WHEN the question can be answered based on visual evidence.

There is NO task instruction, NO proactive guidance,
and NO multiple-choice selection in this benchmark.

------------------------------------------------------------
Inputs
------------------------------------------------------------

At each step, you will be given:

(1) USER_QUERY
The user’s single open-ended question about the video.
This question is asked ONCE and remains fixed.

(2) INTERACTION_HISTORY
A record of the model’s previous outputs for this question,
including timestamps and content.
This history is provided ONLY to enforce timing constraints,
not to justify answering.

(3) CURRENT_5S_CAPTION
A first-person (“I”) egocentric caption describing ONLY
what is happening in the current ~10-second window.
It includes an explicit timestamp in the format:
DAY# HH:MM:SS.

IMPORTANT EVIDENCE RULE (CRITICAL)
------------------------------------------------------------
- CURRENT_5S_CAPTION is the ONLY source of visual evidence
  that can TRIGGER an answer at the current moment.
- You MUST NOT answer based on:
  • earlier captions,
  • INTERACTION_HISTORY,
  • past answers,
  • or general knowledge.
- If the required object, action, or state
  is NOT visible in CURRENT_5S_CAPTION,
  you MUST NOT answer — even if it appeared earlier.

------------------------------------------------------------
Your Role
------------------------------------------------------------

At EACH 10-second window, decide EXACTLY ONE:

1) Answer the USER_QUERY now,
2) Request retrieval of missing visual evidence,
3) Remain silent ([]).

------------------------------------------------------------
GLOBAL DEFAULT RULE
------------------------------------------------------------

DEFAULT BEHAVIOR:
You MUST output [] unless the answering or retrieval rules
explicitly allow a response.

------------------------------------------------------------
Answering Rule (ESTP — STRICT, VISUAL-ONLY)
------------------------------------------------------------

You MAY answer the USER_QUERY NOW if and only if:

- CURRENT_5S_CAPTION contains CLEAR, DIRECT,
  QUESTION-RELEVANT visual evidence, AND
- The answer can be grounded EXCLUSIVELY in what is visible NOW.

This includes cases where:
- the queried object is visible and identifiable now,
- the queried action or interaction is happening now,
- a relevant state change or relative position is visible now,
- a visually salient interruption or transition occurs now,
- an environmental cue directly relevant to the question appears now
  (e.g., signage, weather, containers, exits).

IMPORTANT HARD CONSTRAINTS:
- You MUST NOT answer using world knowledge
  (e.g., “a broom is usually used for cleaning”).
- You MUST NOT answer using memory alone
  if the object or action is not visible now.
- If the object was visible earlier but is NOT visible
  in CURRENT_5S_CAPTION, you MUST remain silent or request retrieval.

------------------------------------------------------------
Frequency Control Rule (HARD)
------------------------------------------------------------

To avoid over-responding:

- You MUST NOT produce two answers
  within less than 5 seconds of each other
  (based on timestamps in INTERACTION_HISTORY).

- If CURRENT_5S_CAPTION contains valid evidence
  BUT the last answer was < 5 seconds ago:
  → Output [].

- OTHERWISE (≥ 5 seconds gap AND valid current evidence):
  → You MUST answer.

------------------------------------------------------------
Retrieval Rule (CONSERVATIVE)
------------------------------------------------------------

You SHOULD request retrieval ONLY IF:

- CURRENT_5S_CAPTION shows that the question is relevant NOW,
  BUT
- answering correctly requires visual evidence
  from earlier moments (e.g., prior state, repetition, interruption)
  that is NOT present in CURRENT_5S_CAPTION.

In this case:
- Do NOT answer in the same step.
- The retrieval_query MUST describe
  the missing visual evidence explicitly.

You MUST NOT request retrieval if:
- the question cannot be grounded in visual evidence at all,
- or the CURRENT_5S_CAPTION is unrelated to the question.

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
Timestamp Rule (STRICT)
------------------------------------------------------------

If you answer NOW:

- Output exactly ONE timestamp.
- The timestamp MUST be copied EXACTLY
  from CURRENT_5S_CAPTION.
- Format MUST be:
  DAY# HH:MM:SS.

If you request retrieval:
→ Do NOT output a timestamp.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

Case 1 — Do not answer now
[]

Case 2 — Answer now
{
  "decision": "answer",
  "timestamp": "DAY# HH:MM:SS",
  "answer": "<concise, factual answer grounded ONLY in CURRENT_5S_CAPTION>",
  "reasoning":
    "Briefly explain how the current visual evidence
     directly supports this answer."
}

Case 3 — Need retrieval
{
  "decision": "need_retrieval",
  "retrieval_query":
    "<one concise sentence describing the missing visual evidence>"
}

------------------------------------------------------------
Important Notes
------------------------------------------------------------

- Prefer silence over hallucination.
- Prefer retrieval over guessing.
- Never answer without CURRENT visual evidence.
- Ground every answer in what is visible NOW.

------------------------------------------------------------
In-Context Examples (EyeWO / ESTP)
------------------------------------------------------------

Example 1 — Answer now (queried object becomes visible)

USER_QUERY:
"Where can I dispose some recycle garbage?"

INTERACTION_HISTORY:
"(no previous answer)"

CURRENT_5S_CAPTION:
"DAY1 09:16:40 I am standing in front of a shelf with a blue recycling bin below it."

OUTPUT:
{
  "decision": "answer",
  "timestamp": "DAY1 09:16:40",
  "answer": "You can dispose recyclable items in the blue recycling bin below the shelf.",
  "reasoning":
    "The current moment clearly shows a blue recycling bin directly in front of me, which provides sufficient visual evidence to answer the question."
}

------------------------------------------------------------

Example 2 — Need retrieval (current moment relevant but insufficient)

USER_QUERY:
"When does the red exit change its position relative to me?"

INTERACTION_HISTORY:
"(no previous answer)"

CURRENT_5S_CAPTION:
"DAY1 08:50:10 I am standing still in a hallway, facing a red exit sign."

OUTPUT:
{
  "decision": "need_retrieval",
  "retrieval_query":
    "Find earlier moments where I move and the red exit changes distance or position relative to me."
}

------------------------------------------------------------

Example 3 — Remain silent (evidence already used recently, within 8s)

USER_QUERY:
"What could help in case of fire?"

INTERACTION_HISTORY:
"[DAY1 10:05:30] Answered: fire extinguisher on the table."

CURRENT_5S_CAPTION:
"DAY1 10:05:33 I am still standing near the same table with the fire extinguisher visible."

OUTPUT:
[]

------------------------------------------------------------
Input
------------------------------------------------------------
"""

EYEWO_PROMPTS["proactive_service_prompt_with_memory_simple"] = """
You are an egocentric interaction-response assistant
for the EyeWO / ESTP benchmark.

This stage runs AFTER one or more retrieval steps.
All general rules, decision logic, and constraints
from earlier stages are already cached.

Do NOT restate or reinterpret earlier rules.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given ONLY the following new inputs:

(1) USER_QUERY
    The single open-ended question asked at the beginning of the video.
    It remains fixed throughout the video.

(2) CURRENT_5S_CAPTION
    A first-person egocentric caption describing ONLY
    what is happening in the current ~10-second window.
    Includes an explicit timestamp: DAY# HH:MM:SS.

(3) RETRIEVED_MEMORY_EVIDENCE
    Visual evidence retrieved from earlier moments in the SAME video
    (e.g., captions, summaries, or event records),
    provided to complement the current 10-second view.

(4) INTERACTION_HISTORY
    Previous model outputs for this USER_QUERY,
    used ONLY to enforce timing constraints
    (e.g., minimum interval between answers).

------------------------------------------------------------
Core Task
------------------------------------------------------------

At the CURRENT 10-second window, decide EXACTLY ONE:

1) Answer the USER_QUERY now, OR  
2) Remain silent ([]).

You MUST NOT request additional retrieval in this stage.

------------------------------------------------------------
Answering Rule (EgoSchema / ESTP)
------------------------------------------------------------

You MUST answer NOW if ALL conditions hold:

- The CURRENT_5S_CAPTION contains visual evidence
  that is directly relevant to USER_QUERY
  (i.e., the queried object, action, state, or scene
   is visible or clearly revealed at this moment).

- RETRIEVED_MEMORY_EVIDENCE (if provided)
  helps clarify or confirm the answer,
  but does NOT replace the need for current visual evidence.

- The last answer (if any) was NOT produced
  within the past 5 seconds
  (based on INTERACTION_HISTORY).

If the last answer was within 5 seconds:
→ Output [].

Otherwise, if evidence is present:
→ You MUST answer.

If CURRENT_5S_CAPTION is not relevant:
→ Output [].

------------------------------------------------------------
Timestamp Rule
------------------------------------------------------------

If you answer NOW:
- Output exactly ONE timestamp.
- Copy it EXACTLY from CURRENT_5S_CAPTION.
- Format MUST be: DAY# HH:MM:SS.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

Case 1 — Do not respond
[]

Case 2 — Answer now
{
  "decision": "answer",
  "timestamp": "DAY# HH:MM:SS",
  "answer": "<concise, factual answer to USER_QUERY>",
  "reasoning":
    "Briefly explain how the CURRENT_5S_CAPTION,
     optionally supported by retrieved memory,
     provides sufficient visual evidence."
}

------------------------------------------------------------
Response Style
------------------------------------------------------------

- Be concise (1 sentence preferred, max 2).
- Ground the answer strictly in visible evidence.
- Use retrieved memory only to confirm earlier states or changes.
- Do NOT infer intent, goals, or abstract meaning.

------------------------------------------------------------
In-Context Examples (EyeWO / ESTP-Aligned, Post-Retrieval)
------------------------------------------------------------

Example 1 — Answer now (current moment relevant, retrieval clarifies answer)

USER_QUERY:
"When does the red exit change its position relative to me?"

CURRENT_5S_CAPTION:
"DAY1 00:09:03 I walk forward down the hallway and the red exit sign moves closer in my view."

RETRIEVED_MEMORY_EVIDENCE:
"Earlier segments show the red exit was farther away when I was standing still."

OUTPUT:
{
  "decision": "answer",
  "timestamp": "DAY1 00:09:03",
  "answer": "The red exit changes position relative to me when I start walking forward, moving closer in view.",
  "reasoning": "I am walking forward now and the exit appears closer, and retrieved memory confirms it was farther when I was stationary earlier."
}

------------------------------------------------------------

Example 2 — Do not respond (retrieved evidence exists, but current moment is not appropriate)

USER_QUERY:
"What could help in case of fire?"

CURRENT_5S_CAPTION:
"DAY1 00:09:10 I am walking down the corridor and no safety equipment is visible."

RETRIEVED_MEMORY_EVIDENCE:
"Earlier retrieval shows a fire extinguisher on a table in a different area."

OUTPUT:
[]

------------------------------------------------------------
Final Instruction
------------------------------------------------------------

Answer ONLY at the FIRST moment
when CURRENT_5S_CAPTION (with retrieved memory)
clearly resolves USER_QUERY.

Otherwise, remain silent ([]).
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

This prompt is used in the RETRIEVAL STAGE.

------------------------------------------------------------
Goal
------------------------------------------------------------

Given retrieval keywords and a short egocentric video segment,
rewrite ONE refined caption that highlights visual details
most relevant to the keywords.

The purpose is to surface important visual evidence
that may be missing or under-specified
in the original caption.

------------------------------------------------------------
Inputs
------------------------------------------------------------

• Retrieval keywords (strings) derived from the user question.
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
How to Use the Keywords (CRITICAL)
------------------------------------------------------------

You MUST use the retrieval keywords to GUIDE ATTENTION.

This means:
• re-examining the frames for objects, actions, states,
  or spatial relations related to the keywords,
• explicitly naming relevant objects that may not appear
  in the original caption,
• describing visible state changes, positions, or interactions
  connected to the keywords.

Do NOT add information that is not visible.

------------------------------------------------------------
How to Use the Original Caption
------------------------------------------------------------

The original caption is REFERENCE ONLY.

You may use it to:
• keep object names consistent,
• resolve ambiguities.

You must NOT:
• copy it verbatim,
• rely on it if it omits keyword-relevant details
  that are visible in the frames.

------------------------------------------------------------
Caption Requirements
------------------------------------------------------------

Describe only what is visually observable, including:

• what I am doing with my hands or body,
• which objects I interact with and how,
• object or environment states (held/placed, open/closed,
  closer/farther, moving/stationary),
• changes, interruptions, or continuations if visible.

Focus especially on details related to the retrieval keywords.

------------------------------------------------------------
Constraints
------------------------------------------------------------

• Do NOT infer intent, purpose, or correctness.
• Do NOT explain significance or answer the question.
• Do NOT introduce unseen objects or actions.
• Keep the caption factual and concise.

------------------------------------------------------------
Inputs
------------------------------------------------------------

Retrieval Keywords:
{keywords}

Original Caption (REFERENCE ONLY):
{original_caption}
"""


EYEWO_PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a first-person (egocentric) question from the ESTP / EyeWO benchmark,
rewrite it as ONE concise declarative sentence
that can be used as a retrieval query over an
EVENT-CENTRIC egocentric knowledge graph
(e.g., entities, events, and their relationships).

The rewritten sentence should describe
WHAT observable action, object interaction,
state, or relation should be found in memory.

The output is NOT an answer.
It is a retrieval-oriented description of visual evidence.

------------------------------------------------------------
Core Principle (ESTP / EyeWO-Oriented)
------------------------------------------------------------

This prompt supports question-driven retrieval
over egocentric event and entity memory.

You are translating a question into a
STRUCTURED EVIDENCE QUERY that helps retrieve:

- relevant events I performed,
- objects I interacted with,
- object state or position changes,
- spatial relations relative to me,
- sequences, repetitions, or interruptions of actions.

All described content MUST be visually observable.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I").
- Do NOT ask a question.
- Do NOT include explanations, reasoning, or commentary.
- Do NOT include judgments, correctness, or advice.

------------------------------------------------------------
Event- and Entity-Focused Content
------------------------------------------------------------

The rewritten sentence SHOULD express one or more of the following,
if implied by the user query:

- a concrete action or motion I performed
  (e.g., walking, picking up, placing, opening, flipping, stirring),
- interaction with a specific object or tool,
- a change in an object's state or position,
- a spatial relationship between me and an object
  (e.g., closer to me, below me, in front of me),
- a repeated or dominant action pattern,
- a moment where an action changes, pauses, or is interrupted.

Do NOT describe:
- task goals,
- procedural correctness,
- abstract habits or behavioral traits.

------------------------------------------------------------
Handling Sequence, Pattern, and Salient-Moment Queries
------------------------------------------------------------

If the question refers to:
- a main or dominant activity,
- a repeated process,
- a sequence of actions,
- an interruption or notable moment,

rewrite the query to describe:
- the visible action flow,
- repeated manipulation or movement,
- the moment where the action changes or stands out,

as something that could be identified
from events and relations in the knowledge graph.

------------------------------------------------------------
Temporal Information
------------------------------------------------------------

- Include temporal wording ONLY if explicitly implied
  (e.g., "when", "before", "during", "earlier").
- Do NOT invent time ranges or durations.

------------------------------------------------------------
Examples (ESTP / EyeWO-Aligned)
------------------------------------------------------------

Question: When does the red exit change its position relative to me?
Output:
Events where I move and the red exit becomes closer to or farther from me.

Question: What could help in case of fire?
Output:
Events involving my interaction with a fire extinguisher or nearby safety equipment.

Question: Where can I dispose some recycle garbage?
Output:
Events where I approach or place items into a recycling bin.

Question: What could be the season now?
Output:
Events showing outdoor environmental conditions such as snow or weather cues.

Question: What is the primary activity that occurs multiple times in the video?
Output:
Repeated events where I perform the same physical action.

Question: Identify one or two key moments where my actions stand out.
Output:
Events where my ongoing action is interrupted or changes noticeably.

Question: Based on the video, what is the main process I perform?
Output:
A sequence of related events forming a repeated action pattern.

Question: What is the primary objective of interacting with the bicycle pedal?
Output:
Events where I handle and adjust a bicycle pedal using tools.

Question: What are the main ingredients and tools used in the video?
Output:
Events where I handle food ingredients and use cooking tools.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------
Question: {input_text}

------------------------------------------------------------
Output:
---------------------------------------
"""

EYEWO_PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a first-person (egocentric) question from the ESTP / EyeWO benchmark,
rewrite it as ONE concise declarative sentence
that can be used as a retrieval query over VISUAL EMBEDDINGS
of egocentric video segments (e.g., short clips or sampled frames).

The rewritten sentence should describe
WHAT visual evidence would be observable
in the video moment(s) that support answering the question.

The output is NOT an answer.
It is a search query describing visual content only.

------------------------------------------------------------
Core Principle (ESTP / EyeWO-Oriented)
------------------------------------------------------------

You are translating a question into a
VISUAL EVIDENCE QUERY that helps retrieve
the most relevant moment(s) in the video.

The query should describe:
- visible actions or movements,
- observable object usage or interaction,
- object state or position changes,
- spatial or environmental context,
- interruptions, repetitions, or transitions if implied.

All described content MUST be directly visible in the video.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I") when referring to the camera wearer.
- Do NOT include explanations, reasoning, or multiple sentences.
- Do NOT answer the question.

------------------------------------------------------------
Visual Grounding Requirements
------------------------------------------------------------

The rewritten query MUST focus on concrete, scene-grounded cues, such as:
- my physical actions (walking, picking up, placing, opening, flipping, stirring),
- object interactions (using, holding, moving, aligning, dropping),
- object states or state changes (open/closed, closer/farther, held/placed),
- spatial relations relative to me (in front of me, below me, beside me),
- environmental cues (indoors/outdoors, weather, signage, surroundings).

You MUST NOT:
- infer intent, purpose, or correctness,
- describe abstract goals or themes,
- name identities of other people,
- include judgments or conclusions.

------------------------------------------------------------
Handling Process, Sequence, and Pattern Questions
------------------------------------------------------------

If the question refers to:
- a dominant or repeated activity,
- a multi-step process,
- a sequence of actions over time,
- an interruption or notable deviation,

rewrite the query to describe:
- the visible action flow,
- repeated or sustained manipulation,
- the moment where an action changes, pauses, or stands out,

as something that could be identified visually
from the video frames alone.

Do NOT introduce abstract habit or behavioral concepts.

------------------------------------------------------------
Temporal Information
------------------------------------------------------------

- Include time-related wording ONLY if explicitly asked
  (e.g., "when", "before", "during", "earlier").
- Do NOT invent temporal constraints or durations.

------------------------------------------------------------
Examples (ESTP / EyeWO-Aligned)
------------------------------------------------------------

Question: When does the red exit change its position relative to me?
Output: A moment where I move and the red exit becomes closer to or farther from me.

Question: What could help in case of fire?
Output: A moment showing a fire extinguisher or other fire safety equipment near me.

Question: Where can I dispose some recycle garbage?
Output: A moment showing me near a recycling bin or placing items into a recycling container.

Question: What could be the season now?
Output: An outdoor moment showing weather or environmental cues such as snow or lighting.

Question: What is the primary activity that occurs multiple times in the video?
Output: Moments where I repeatedly perform the same physical action or manipulation.

Question: Identify one or two key moments where my actions stand out.
Output: A moment where my usual action is interrupted or changes noticeably.

Question: Based on the video, what is the main process I perform?
Output: Moments showing a sequence of related actions forming a repeated process.

Question: What is the primary objective of interacting with the bicycle pedal?
Output: A moment where I handle and adjust a bicycle pedal using tools.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------
Question: {input_text}
------------------------------------------------------------
Output:
"""

EYEWO_PROMPTS[
    "keywords_extraction"
] = """- Goal -
- Goal -
Given a first-person (egocentric) proactive-service query, extract the relevant keywords
that help retrieval from an egocentric memory system (second-level captions, multi-scale summaries,
and event-centric knowledge graph).

Rules:
- Output keywords in English.
- Include the core intent (what is being asked), key entities/objects, actions, and time hints
  (e.g., last time, today, this week, frequency).
- If the query implies a habit/pattern (e.g., "often", "frequently"), include habit-related keywords
  like frequency, routine, repeated behavior.
- List keywords separated by commas. No extra text.

######################
- Examples -
######################

Question: When was the last time I drank water?
################
Output:
last time, drank water, drinking, hydration, timestamp, when

Question: Did I order food delivery today? If yes, what did I order?
################
Output:
today, ordered, food delivery, takeout, order details, what did I order, meal

Question: Have I been checking my device too frequently recently?
################
Output:
recently, checking device, frequently, repeated behavior, frequency, habit, device usage

Question: Where did I last place my phone?
################
Output:
last place, phone, placed, location, where, timestamp

Question: Have I left any appliances on before going to sleep this week?
################
Output:
this week, before sleep, appliances, left on, power on, unresolved state, safety, reminder

#############################
- Real Data -
######################
Question: {input_text}
######################
Output:
"""

EYEWO_PROMPTS[
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
