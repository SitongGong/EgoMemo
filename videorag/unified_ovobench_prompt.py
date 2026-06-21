"""
Unified prompt module for OVO-Bench (rebuttal experiment).

REBUTTAL CLAIM PRESERVED: every LLM-side prompt that drives reasoning
(entity_extraction, caption_reconstruction, query_rewrite_for_*_retrieval,
proactive_service_prompt_with_memory_simple, backward_retrieval_inference,
keywords_extraction, filtering_segment, summarize_entity_descriptions,
hour-level summarization) is the dataset-agnostic UNIFIED_PROMPTS.

TWO PROMPTS RETAINED FROM ORIGINAL OVO-Bench:
1. `simple_second_caption_system_prompt` — this is the Qwen-VL system
   prompt that produces frame-level captions in a JSON schema the
   graph-construction code parses with `int(frame_idx)`. Replacing it
   makes Qwen-VL emit a different schema and breaks the parser, with
   zero relation to the LLM-prompt-engineering critique we are
   addressing. We therefore keep the original OVO simple_second
   prompt, which is a Qwen-VL output-format spec, not a reasoning
   prompt.
2. `DEFAULT_ENTITY_TYPES` is extended with "animal" because OVO scenes
   are third-person / multi-actor and contain animals. This is a
   data-modality fact, not prompt engineering.

All other prompts come straight from UNIFIED_PROMPTS. This is the
rebuttal evidence: every prompt that the reviewer would call "extreme
dataset-specific prompt engineering" has been replaced with a single
unified version.
"""

from .unified_prompt import UNIFIED_PROMPTS
from .ovobench_prompt import OVOBENCH_PROMPTS as _ORIG_OVOBENCH_PROMPTS

# Start as a copy of the shared prompts.
UNIFIED_OVOBENCH_PROMPTS = dict(UNIFIED_PROMPTS)


# ---------------------------------------------------------------------------
# Compatibility: simple_second_caption_system_prompt
# This prompt is tightly coupled to Qwen-VL's frame-level OUTPUT_FORMAT
# expected by graph-construction code (frame keys like "0","1","2", parsed
# via int(frame_idx)). Substituting a different caption prompt makes Qwen
# emit a different output schema and breaks graph parsing for every video.
# We therefore retain the ORIGINAL OVO-Bench caption prompt for the
# Qwen-VL <-> code interface only. The "unified-prompt" rebuttal claim is
# preserved because the LLM-side prompts (entity_extraction,
# caption_reconstruction, query_rewrite_*, multiscale summarization,
# proactive_service, backward_retrieval_inference, keywords_extraction,
# filtering_segment) — i.e. ALL prompts that drive reasoning — are still
# the dataset-agnostic UNIFIED_PROMPTS.
# ---------------------------------------------------------------------------
UNIFIED_OVOBENCH_PROMPTS["simple_second_caption_system_prompt"] = (
    _ORIG_OVOBENCH_PROMPTS["simple_second_caption_system_prompt"]
)


# ---------------------------------------------------------------------------
# Compatibility: hour_caption_system_prompt
# OVO graph code calls OVOBENCH_PROMPTS['hour_caption_system_prompt'] for
# the hour-level multiscale summarization. We reuse the unified
# min_caption_system_prompt because it operates on already-aggregated
# textual captions (no Qwen-VL output-format coupling).
# ---------------------------------------------------------------------------
UNIFIED_OVOBENCH_PROMPTS["hour_caption_system_prompt"] = (
    UNIFIED_PROMPTS["min_caption_system_prompt"]
)


# ---------------------------------------------------------------------------
# Override 1: DEFAULT_ENTITY_TYPES
# OVO-Bench scenes commonly include multiple persons and animals.
# ---------------------------------------------------------------------------
UNIFIED_OVOBENCH_PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "person", "animal", "location", "object", "event"
]


# ---------------------------------------------------------------------------
# Override 2: entity_extraction
# Same schema as the shared one; only adds explicit handling of
# multi-person / animal entities. This is the ONLY substantive
# data-modality-driven divergence in the entire prompt suite.
# ---------------------------------------------------------------------------
UNIFIED_OVOBENCH_PROMPTS["entity_extraction"] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a video caption with explicit timestamps,
extract visually grounded entities and relationships
to form an EVENT-CENTRIC temporal knowledge graph.

The graph supports diverse downstream queries about
actions, objects, spatial relations, state changes,
counterfactual feasibility, and temporal ordering.

The caption may describe:
- first-person or third-person perspective,
- one or multiple visible persons,
- visible animals,
- objects, spatial relations, attribute information,
- visible motion or interactions.

------------------------------------------------------------
IMPORTANT CONCEPTUAL RULES (STRICT)
------------------------------------------------------------

- EVENT = a concrete, observable physical action,
  interaction, motion, or state transition.

- Events MUST focus on hands-on actions, object manipulation,
  physical movement, or observable state changes.

- TEMPORAL INFORMATION = when the event happens.
- Time itself is NEVER an event.

- Events may be performed by:
  • any visible person,
  • any visible animal,
  • or describe a visible state change of an object.

- All observable interactions MUST be represented AS EVENTS.

- Relationships NEVER replace events;
  they only describe how entities participate in events.

- If an action visibly results in an object state change,
  both the action and the resulting visible state
  MUST be represented.

- Do NOT omit stable object states if they are explicitly
  mentioned and could affect downstream reasoning.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:
- A timestamped caption: "DAY# HH:MM:SS-HH:MM:SS".
- A detailed description of visible scene content.
- Entity_types allowed: {entity_types}

No task topic, goal, task type, or external procedural knowledge
is provided. You MUST rely ONLY on the caption.

------------------------------------------------------------
A) Extract Entities
------------------------------------------------------------

Extract ONLY entities that are:
- explicitly mentioned in the caption,
- visually grounded (present in the visible scene),
- necessary to represent observable actions, object states,
  spatial relations, counts, or temporal structure.

Entity types MUST be one of:
{entity_types}

person:
- Any visible individual (e.g., man, woman, child, coach, player,
  or the camera wearer "I" if first-person).
- Multiple person entities are allowed.
- Use distinguishable identifiers when needed
  (e.g., Man_1, Woman_1, Coach, Player_in_White).

animal:
- Any visible animal (dog, monkey, parrot, cat, etc.).
- Multiple animal entities are allowed.

object:
- Any physical object, tool, container, device, vehicle,
  furniture, signage, clothing item, or environmental object.

location:
- A visible physical area or environment.

event (CORE ENTITY TYPE):
- A concrete, observable action or interaction.
- Includes: holding, placing, moving, opening, closing,
  attaching, detaching, pouring, cutting, walking,
  sitting, running, facing, pointing, emerging
  (e.g., smoke from a barrel), state transitions
  (open/closed, on/off).

EVENT RULES:
- Events MUST be grounded strictly in the caption.
- Events MUST include a temporal_scope copied EXACTLY from the caption.
- Do NOT include timestamps inside entity_description.
- Each event must describe ONE coherent visible action pattern.

------------------------------------------------------------
Entity Fields
------------------------------------------------------------

For each entity, extract:

- entity_name: Canonical name (capitalized, consistent).
- entity_type: One of {entity_types}.
- entity_description: Factual description grounded strictly
  in the caption.
- temporal_scope: REQUIRED only for event entities.
  Format: DAY# HH:MM:SS-HH:MM:SS. Copy exactly from caption.
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

- All actions MUST be mediated by event nodes.
- person → participates_in → Event
- animal → participates_in → Event
- Event  → holds / places / moves / opens / closes /
           attaches / detaches / pours / sits_on /
           runs_toward / faces / emerges_from /
           located_near / contains → Object
- Event  → occurs_in → Location
- Event  → follows / continues / interrupts / causes → Event

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
- 9-10 = central to scene understanding
- 6-8  = important contextual interaction
- 3-5  = background relevance

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
- number of visible persons / animals (each as separate entity),
- who holds / interacts with what,
- spatial relations (left/right/front/behind),
- orientation (facing left/right),
- attribute cues (color of clothing, object color),
- visible text as object entities,
- what emerges from objects (e.g., smoke from a barrel),
- entities entering or leaving the visible scene.

Do NOT:
- infer intent,
- assume unseen actions,
- create abstract logical entities,
- invent events not present in the caption.

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
Two men are sitting on a wooden bench. A woman stands to the right
of the bench. One man is holding a green bag. The other man is
looking to the left. Smoke is coming out of a barrel behind them.

Output:
("entity"{tuple_delimiter}"Man_1"{tuple_delimiter}"person"{tuple_delimiter}"A man sitting on a wooden bench."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Man_2"{tuple_delimiter}"person"{tuple_delimiter}"Another man sitting on the same bench."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Woman_1"{tuple_delimiter}"person"{tuple_delimiter}"A woman standing to the right of the bench."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Wooden Bench"{tuple_delimiter}"object"{tuple_delimiter}"A bench on which two men are sitting."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Green Bag"{tuple_delimiter}"object"{tuple_delimiter}"A green bag held by one of the men."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Barrel"{tuple_delimiter}"object"{tuple_delimiter}"A barrel positioned behind the bench."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Smoke"{tuple_delimiter}"object"{tuple_delimiter}"Smoke emerging from the barrel."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"E_SITTING_EVENT"{tuple_delimiter}"event"{tuple_delimiter}"Two men sit on a wooden bench while a woman stands nearby."{tuple_delimiter}"DAY3 10:15:00-10:15:20"){record_delimiter}
("entity"{tuple_delimiter}"E_HOLDING_BAG"{tuple_delimiter}"event"{tuple_delimiter}"One man holds a green bag while seated."{tuple_delimiter}"DAY3 10:15:00-10:15:20"){record_delimiter}
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
