"""
Unified prompt module for the rebuttal experiment.

Goal: Demonstrate that EgoMemo's performance does NOT depend on
dataset-specific prompt engineering. We provide ONE shared set of prompts
that is used identically across:
  - EgoSchema (multiple-choice long-form QA)
  - EgoTaskQA (multiple-choice procedural QA)
  - QAEgo4D (open-ended episodic memory QA)
  - OVO-Bench backward (long-form video grounding)

The only key that retains a dataset-modality-aware variant is
`entity_extraction`, because OVO-Bench is third-person/multi-actor whereas
the other three are strictly first-person egocentric. The
`entity_extraction` for OVO is provided in `unified_ovobench_prompt.py`
and is structurally identical (same fields, same output format) - it only
adds support for multiple-person/animal entities.

Reference:
 - Prompts are derived from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
UNIFIED_PROMPTS = {}


# ---------------------------------------------------------------------------
# 1. Caption generation prompt (shared by all benchmarks)
#    Replaces: caption_system_prompt_with_query / simple_second_caption_system_prompt
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["caption_system_prompt_with_query"] = """
You are a video episodic frame recorder.

You are given a short video segment sampled from a longer video.
The segment may be one frame or several sequential frames.

Each segment is an independent EVIDENCE UNIT.
Your job is to record ALL observable visual evidence
that may later support downstream reasoning, including:

• object identity, attributes, counting,
• spatial relations and orientation,
• action and interaction sequences,
• object state tracking and state transitions,
• causal links between actions and visible effects,
• temporal ordering and transitions.

You are NOT answering any question.
You are NOT inferring hidden intentions.
You are ONLY recording visible evidence.

------------------------------------------------------------
QUESTION-AWARE RECORDING (ATTENTION GUIDANCE ONLY)
------------------------------------------------------------

QUESTIONS FOR THIS VIDEO (REFERENCE ONLY):
{question}

IMPORTANT:

• There may be one or multiple questions for this video.
• Some questions may include multiple answer options.
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
• Caption coverage must integrate evidence
  that could support OR contradict ANY option
  from ANY question.

This caption must function as a neutral,
maximally complete visual evidence record
covering ALL provided questions simultaneously.

------------------------------------------------------------
ANTI-ABSTRACTION RULE (MANDATORY)
------------------------------------------------------------

You MUST NEVER use vague words such as:
"something", "object", "item", "tool", "device",
"person", "somebody", "someone".

You MUST ALWAYS replace abstract references
with the most specific visible noun possible
(e.g., "red plastic cup", "metal wrench", "white bucket",
"man in black jacket").

If a question uses abstract terms,
YOU must resolve them into concrete visible entities
based strictly on what is seen in the frames.

------------------------------------------------------------
CORE RECORDING OBJECTIVES
------------------------------------------------------------

Your caption must explicitly support:

1) ENTITY IDENTITY & COUNT
   - Record every visible person, animal, object,
     and significant background element.
   - If multiple entities of the same kind are present,
     record the count and distinguishing features.

2) OBJECT STATE TRACKING
   - Record the visible state of each object:
     open/closed, on/off, attached/detached,
     filled/empty, broken/intact, visible/hidden.
   - If a state changes in this segment,
     describe BEFORE and AFTER.

3) ACTION → STATE CAUSAL LINKS
   - If an action visibly changes an object's state,
     explicitly describe: action → resulting visible state.
   - If no state change occurs, explicitly state stability.

4) SPATIAL RELATIONS & ORIENTATION
   - Record positions (left/right/front/behind/inside/on top of).
   - Record where each entity faces.
   - Record what is visible in foreground vs background.

5) TEMPORAL ORDER & TRANSITIONS
   Explicitly indicate:
     • continuation, transition, interruption,
       action completion, new action begins.
   - Maintain clear chronological ordering.
   - Separate each action and its visible consequences.
   - Do NOT merge multiple steps into one description.

6) MULTI-OBJECT & MULTI-ACTOR CHAINS
   - If multiple entities undergo changes or interactions,
     record each separately.
   - Preserve the logical sequence of changes.

7) TEXT & LABELS
   - Record ALL visible text exactly as written.
   - Include brand names, labels, instructions, symbols.

------------------------------------------------------------
FRAME-DEPENDENT RECORDING RULE
------------------------------------------------------------

The number of frames in the input segment is NOT fixed.

If only a single frame is provided:
- Record all visible entities, states, spatial relations,
  and ongoing actions.
- Do NOT fabricate temporal transitions or state changes.
- Only describe state changes if they are visually evident
  within the frame (e.g., motion blur, partially completed action).

If multiple frames are provided:
- Explicitly describe temporal transitions across frames.
- Record BEFORE and AFTER states when changes are visible.
- Maintain clear chronological ordering.

Caption density and temporal reasoning must strictly match
the actual number of visible frames. Do NOT assume a fixed duration.

------------------------------------------------------------
GLOBAL EVIDENCE CAPTION REQUIREMENT
------------------------------------------------------------

After detailed frame-wise recording,
produce ONE global evidence caption.

This caption must:

• Enumerate:
    - all entities present,
    - all actions performed,
    - all object state changes,
    - all stable object states,
    - causal links between actions and visible effects.
• Explicitly state whether the segment shows:
    continuation, transition, interruption, or completion.
• Maintain chronological clarity.

Preferred length: dense but complete (2-4 sentences).

This caption functions as:
A structured evidence record for downstream reasoning.

------------------------------------------------------------
Output JSON Format (STRICT)
------------------------------------------------------------
{output_format}

------------------------------------------------------------
FINAL RULES
------------------------------------------------------------

If an object changes state,
failing to describe its previous and current visible state is an error.

If an action begins,
failing to describe its visible preconditions is an error.

If a question uses abstract wording,
failing to replace it with concrete visible entities is an error.

Maximal factual coverage is required.
No abstraction. No hallucination.
No interpretation beyond visible evidence.
"""


# ---------------------------------------------------------------------------
# 2. Activity-level (1-min) summarization prompt (shared)
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["min_caption_system_prompt"] = """
You are an activity-level summarizer for video memory construction.

You will be given a sequence of short clip-level captions
covering approximately one minute of video, each with a timestamp.

Your task is to produce ONE concise activity-level caption
that summarizes what happened in this minute.

------------------------------------------------------------
INPUT FORMAT
------------------------------------------------------------

A list of timestamped clip-level captions:
- "DAY# HH:MM:SS-HH:MM:SS: <clip caption text>"
- Captions appear in chronological order.

------------------------------------------------------------
OUTPUT REQUIREMENTS
------------------------------------------------------------

Produce a single 2-3 sentence summary that:

1) Identifies the main activity or theme of this minute
   (e.g., "preparing ingredients", "examining a device",
    "walking through a room", "interacting with another person").

2) Lists key entities involved (people, animals, objects, locations).

3) Indicates progression: what was started, completed,
   continued, or interrupted within this minute.

4) Preserves any visible state changes that may matter later
   (e.g., "the box was opened", "the engine was started").

5) Preserves the timestamp range:
   begin with "DAY# HH:MM:SS-HH:MM:SS: ".

------------------------------------------------------------
STRICT RULES
------------------------------------------------------------

• Do NOT introduce information not present in the clip captions.
• Do NOT speculate about intent or future actions.
• Do NOT use vague abstractions ("something", "an object").
• If multiple distinct activities occur, mention each briefly.
• Use natural English. No bullet lists.

------------------------------------------------------------
Output Format
------------------------------------------------------------

DAY# HH:MM:SS-HH:MM:SS: <2-3 sentence activity summary>
"""


# ---------------------------------------------------------------------------
# 3. Entity extraction prompt (shared by ego-centric benchmarks).
#    OVO-Bench has its own variant in unified_ovobench_prompt.py that adds
#    multi-actor / animal support — structurally identical.
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["entity_extraction"] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a video caption with explicit timestamps,
extract visually grounded entities and relationships
to form an EVENT-CENTRIC temporal knowledge graph.

The graph supports diverse downstream queries about
actions, objects, spatial relations, state changes,
counterfactual feasibility, and temporal ordering.

The camera wearer ("I") is the central reference for
egocentric content; for non-egocentric content the
extraction follows the same schema applied to all visible actors.

------------------------------------------------------------
IMPORTANT CONCEPTUAL RULES (STRICT)
------------------------------------------------------------

- EVENT = a concrete, observable physical action,
  interaction, motion, or state transition.

- Events MUST focus on hands-on actions, object manipulation,
  physical movement, or observable state changes.

- TEMPORAL INFORMATION = when the event happens.
- Time itself is NEVER an event.

- All observable interactions MUST be represented AS EVENTS.

- Relationships NEVER replace events;
  they only describe how entities participate in events.

- If an action visibly results in an object state change,
  both the action and the resulting visible state
  MUST be represented.

- If an action begins and a prior visible object state
  is described, that prior state MUST be preserved
  when necessary for reasoning.

- Do NOT omit stable object states if they are explicitly
  mentioned and could affect downstream reasoning.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:
- A timestamped caption: "DAY# HH:MM:SS-HH:MM:SS".
- A detailed first-person description of visible scene content.
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
  spatial relations, or temporal structure.

Entity types MUST be one of:
{entity_types}

person:
- Visible individuals, including the camera wearer ("I").
- Multiple person entities are allowed when applicable.

object:
- Any physical object, tool, container, device, vehicle,
  furniture, signage, clothing item, or environmental object.

location:
- A visible physical area or environment.

event (CORE ENTITY TYPE):
- A concrete, observable action or interaction.
- Includes: holding, placing, moving, opening, closing,
  attaching, detaching, pouring, cutting, walking,
  facing, pointing, state transitions (open/closed, on/off).

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
- person  → participates_in → Event
- Event   → holds / places / moves / opens / closes /
            attaches / detaches / pours / cuts / faces /
            located_near / contains → Object
- Event   → occurs_in → Location
- Event   → follows / continues / interrupts / causes → Event

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
- every entity actually involved in an action,
- every observable state change,
- every spatial relation needed for reasoning,
- every temporal transition between events.

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

Entity_types: [person, location, object, event]

Text:
DAY1 11:09:43-11:09:53:
I open a cardboard box on the wooden table, take out a red plastic cup,
and place it next to the white kettle.

Output:
("entity"{tuple_delimiter}"I"{tuple_delimiter}"person"{tuple_delimiter}"The camera wearer performing the actions."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Cardboard Box"{tuple_delimiter}"object"{tuple_delimiter}"A cardboard box on the wooden table that is opened."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Red Plastic Cup"{tuple_delimiter}"object"{tuple_delimiter}"A red plastic cup taken out of the cardboard box."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"White Kettle"{tuple_delimiter}"object"{tuple_delimiter}"A white kettle next to which the cup is placed."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"Wooden Table"{tuple_delimiter}"location"{tuple_delimiter}"The surface on which the actions take place."{tuple_delimiter}"" ){record_delimiter}
("entity"{tuple_delimiter}"E_OPEN_BOX"{tuple_delimiter}"event"{tuple_delimiter}"I open the cardboard box."{tuple_delimiter}"DAY1 11:09:43-11:09:53"){record_delimiter}
("entity"{tuple_delimiter}"E_TAKE_CUP"{tuple_delimiter}"event"{tuple_delimiter}"I take a red plastic cup out of the box."{tuple_delimiter}"DAY1 11:09:43-11:09:53"){record_delimiter}
("entity"{tuple_delimiter}"E_PLACE_CUP"{tuple_delimiter}"event"{tuple_delimiter}"I place the cup next to the white kettle."{tuple_delimiter}"DAY1 11:09:43-11:09:53"){record_delimiter}

("relationship"{tuple_delimiter}"I"{tuple_delimiter}"E_OPEN_BOX"{tuple_delimiter}"participates_in"{tuple_delimiter}"I performed the opening action."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"E_OPEN_BOX"{tuple_delimiter}"Cardboard Box"{tuple_delimiter}"opens"{tuple_delimiter}"The box is the target of the opening action."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"E_TAKE_CUP"{tuple_delimiter}"Red Plastic Cup"{tuple_delimiter}"holds"{tuple_delimiter}"The cup is taken out and held."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"E_PLACE_CUP"{tuple_delimiter}"Red Plastic Cup"{tuple_delimiter}"places"{tuple_delimiter}"The cup is placed next to the kettle."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"E_PLACE_CUP"{tuple_delimiter}"White Kettle"{tuple_delimiter}"located_near"{tuple_delimiter}"The cup ends up next to the kettle."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"E_OPEN_BOX"{tuple_delimiter}"E_TAKE_CUP"{tuple_delimiter}"follows"{tuple_delimiter}"Taking out the cup follows opening the box."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"E_TAKE_CUP"{tuple_delimiter}"E_PLACE_CUP"{tuple_delimiter}"follows"{tuple_delimiter}"Placing the cup follows taking it out."{tuple_delimiter}9){completion_delimiter}

######################
-Input-
Detailed Captions: {input_text}
Entity_types: {entity_types}
######################
Output:
"""


# ---------------------------------------------------------------------------
# 4. Caption reconstruction prompt (shared)
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["caption_reconstruction"] = """
You are a video caption rewriter.

This prompt is used in the RETRIEVAL STAGE
to regenerate a more precise, evidence-focused caption
from a short video segment.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

- Retrieval keywords derived from the user question.
- A short video segment, sampled frames in temporal order.
- An ORIGINAL FINE-GRAINED CAPTION for the same segment
  (generated earlier and grounded in the frames).

Retrieval keywords: {keywords}
Original caption: {original_caption}

------------------------------------------------------------
Your Output
------------------------------------------------------------

- Output EXACTLY ONE rewritten caption.
- Do NOT use JSON or any special formatting.
- The caption MUST be grounded only in what is visible
  in the original caption / frames.

------------------------------------------------------------
Your Role
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

1) Read the keywords. They tell you which visual elements
   should be emphasized (objects, actions, states, environments).

2) Read the original caption. It is your factual baseline.

3) Rewrite ONE caption that:

   - keeps every visible fact from the original caption,
   - reorganizes wording to surface keyword-relevant evidence,
   - resolves abstract references into concrete visible nouns,
   - preserves temporal ordering and state changes,
   - does NOT introduce content not present in the original
     caption / frames.

------------------------------------------------------------
STRICT RULES
------------------------------------------------------------

- Do NOT answer questions.
- Do NOT speculate, infer intent, or generalize.
- Do NOT invent objects, actions, or states.
- Do NOT use vague abstractions ("something", "an object").
- Do NOT compress away details that the keywords mention.

------------------------------------------------------------
Output
------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# 5. Query rewrite for entity / graph retrieval (shared)
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["query_rewrite_for_entity_retrieval"] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a question (and optionally its multiple-choice options)
about a video, rewrite it into EXACTLY ONE concise English
declarative sentence that can retrieve relevant EVENTS,
ENTITIES, and RELATIONS from a video memory system
(event-centric knowledge graph and entity records).

The output is NOT an answer.
It is an EVENT-ORIENTED RETRIEVAL QUERY.

------------------------------------------------------------
Input
------------------------------------------------------------

Question (may include options): {input_text}

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I") if the question is
  about an egocentric video; otherwise keep neutral phrasing.
- Do NOT ask a question.
- Do NOT include explanations.
- Do NOT enumerate the options.
- Do NOT include answer choices in the output.

------------------------------------------------------------
Rewriting Rules
------------------------------------------------------------

1) Mention all key entities (objects, people, locations)
   that appear in the question.
2) Mention key actions or interactions if present.
3) Mention any state, attribute, or temporal cue
   (before/after/first/last) explicitly.
4) Resolve abstract references ("the thing", "it") into
   the concrete entities mentioned in the question
   when possible.
5) Keep the sentence under 40 words.

If options describe multiple candidate entities/actions,
include them as alternatives in the sentence
(e.g., "I want to know whether I picked up the red cup,
the blue mug, or the white plate before sitting down.").

------------------------------------------------------------
Output
------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# 6. Query rewrite for visual retrieval (shared)
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["query_rewrite_for_visual_retrieval"] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a question (and optionally its multiple-choice options)
about a video, rewrite it into a visual-centric description
that can be encoded by a multimodal encoder and matched against
keyframe embeddings in the video memory.

The output is NOT an answer.
It is a VISUAL-CENTRIC RETRIEVAL QUERY focused on what the
relevant frames would look like.

------------------------------------------------------------
Input
------------------------------------------------------------

Question (may include options): {input_text}

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

- Output ONE descriptive sentence (or short paragraph,
  at most 50 words).
- Emphasize VISUAL ATTRIBUTES: objects, colors, shapes,
  positions, postures, scene layout, visible text.
- Do NOT describe abstract intent or motivation.
- Do NOT ask a question.
- Do NOT include explanations or option enumeration.

------------------------------------------------------------
Rewriting Rules
------------------------------------------------------------

1) List the visible entities the question depends on
   (objects, people, locations) with concrete attributes.
2) Describe the scene as it would appear in a frame
   (e.g., "a person holding a red cup near a kitchen sink").
3) Include any temporal cue as a visible state
   (e.g., "before the cup is filled" → "an empty red cup").
4) Avoid abstract terms ("something", "an item");
   use concrete visual nouns.
5) If the question concerns counting, include the count
   directly ("three people standing in a row").

------------------------------------------------------------
Output
------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# 7. Iterative reasoning + retrieval prompt (shared)
#    Replaces: proactive_service_prompt_with_memory_simple
#    Used in egoschema_graph_ablation.py and friends.
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["proactive_service_prompt_with_memory_simple"] = """
You are a video question-answering decision assistant.

This is a TRAINING-FREE, ITERATIVE decision-and-retrieval process.

At each round you must decide EXACTLY ONE action:
1) Answer the question now, OR
2) Request additional retrieval, OR
3) (Only if ROUND_INDEX == MAX_ROUNDS) Output a forced final answer.

You are NOT allowed to guess unless forced by the round limit.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

(1) QUESTION
A natural-language question about a video.

NOTE:
- "c" or similar references in egocentric questions
  always refer to the camera wearer ("I").
- The question may involve:
  • object state before/after actions,
  • causal reasoning, preconditions, counterfactual feasibility,
  • first/last action ordering,
  • visibility/awareness reasoning,
  • spatial relations, counting, attributes, orientation,
  • cross-segment temporal reasoning.

(2) OPTIONS (may be empty)
For multiple-choice tasks, exactly ONE option is correct.
For open-ended tasks, this section is empty and you must
produce a free-form answer.

(3) GLOBAL_CAPTIONS
A sequence of activity-level (1-min) captions with time spans
that summarize different parts of the SAME video.

(4) RETRIEVED_CONTEXT (OPTIONAL)
Additional captions, summaries, or reconstructed segments
retrieved in earlier rounds. May be empty in the first round.

(5) ROUND_INDEX
An integer indicating the current round (starting from 1).

(6) MAX_ROUNDS
The maximum number of allowed rounds.

------------------------------------------------------------
Decision Rules (STRICT)
------------------------------------------------------------

You MUST answer NOW IF AND ONLY IF:
- The combined GLOBAL_CAPTIONS and RETRIEVED_CONTEXT
  contain explicit visual evidence sufficient to determine
  the correct answer (or, for MC, eliminate all but one option),
  AND
- The evidence is consistent and unambiguous.

You MUST request retrieval IF:
- Critical evidence (specific entity, specific moment,
  specific state, specific count) is missing or ambiguous,
  AND
- ROUND_INDEX < MAX_ROUNDS.

In that case, output a precise retrieval query that targets
the specific missing evidence. The query should:
- be a single declarative sentence,
- mention concrete entities/actions/states,
- avoid generic phrasing.

You MUST output a forced final answer IF:
- ROUND_INDEX == MAX_ROUNDS, even if uncertainty remains.
  In this case, choose the most plausible answer based on
  available evidence and explain the residual uncertainty.

------------------------------------------------------------
Output Format (STRICT JSON)
------------------------------------------------------------

If answering (sufficient evidence):
{
  "decision": "answer",
  "answer": "<one short sentence; for MC, the chosen option verbatim>",
  "reasoning": "<one sentence summarizing the supporting evidence>"
}

If forced answer (ROUND_INDEX == MAX_ROUNDS):
{
  "decision": "forced_answer",
  "answer": "<best-guess answer>",
  "reasoning": "<one sentence explaining residual uncertainty>"
}

If requesting retrieval (only when ROUND_INDEX < MAX_ROUNDS):
{
  "decision": "need_retrieval",
  "retrieval_query": "<one declarative sentence describing the missing evidence>",
  "reasoning": "<one sentence explaining what is missing>"
}

------------------------------------------------------------
Style Constraints
------------------------------------------------------------
- One sentence each for "answer" and "reasoning".
- For multiple-choice, the "answer" field must contain
  EXACTLY ONE option (verbatim or the option letter).
- For open-ended, the "answer" field must be a single
  factual sentence.
- Do NOT include extraneous keys.
- Do NOT include markdown.
- Output ONLY the JSON object, nothing else.
"""


# ---------------------------------------------------------------------------
# 8. UNIFIED inference prompt for OVO-Bench backward (rebuttal experiment).
#
#    The original `OVOBENCH_BACKWARD_RETRIEVAL_PROMPT` (hardcoded inside
#    ovobench_retrieval.py) explicitly mentions OVO sub-task labels
#    (EPM / ASI / HLD). The reviewer's complaint is that such per-task
#    prompt engineering accounts for the reported gains. To rebut, we
#    expose this drop-in alternative that:
#      - removes ALL mentions of EPM / ASI / HLD,
#      - removes any reference to the dataset name,
#      - mirrors the iterative answer-or-retrieve protocol used by the
#        egocentric benchmarks (proactive_service_prompt_with_memory_simple).
#
#    Output JSON contract is preserved EXACTLY (decision in
#    answer/forced_answer/need_retrieval; same field names) so the
#    existing parser in `ovobench_retrieval._handle_backward_task`
#    remains compatible without code changes.
#
#    The prompt expects to be wrapped by the same `f"""{PROMPT}\n\n----\n
#    QUESTION...OPTIONS...GLOBAL_CAPTIONS...RETRIEVED_CONTEXT..."""`
#    template that the original code constructs.
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["backward_retrieval_inference"] = """
You are a video question-answering decision assistant.

This is a TRAINING-FREE, ITERATIVE decision-and-retrieval process.
At most TWO retrieval rounds are allowed in this stage.

You are given:
- GLOBAL_CAPTIONS: minute-level summaries covering the entire video.
- RETRIEVED_CONTEXT: additional retrieved captions from previous
  retrieval rounds (may be empty in the first round).

------------------------------------------------------------
Inputs
------------------------------------------------------------

(1) QUESTION
A natural-language question about an earlier portion of the video.

(2) OPTIONS
Multiple-choice options. Exactly ONE is correct.

(3) GLOBAL_CAPTIONS
Activity-level summaries spanning the entire video.

(4) RETRIEVED_CONTEXT (OPTIONAL)
Additional retrieved captions from previous rounds. May be empty.

------------------------------------------------------------
Decision Rules
------------------------------------------------------------

CASE 1 - RETRIEVED_CONTEXT is EMPTY (first round):

You may ANSWER immediately ONLY IF:
- The answer can be clearly determined from GLOBAL_CAPTIONS alone.
- Exactly ONE option is consistent with the evidence.

Otherwise -> request retrieval.

CASE 2 - RETRIEVED_CONTEXT EXISTS (after retrieval):

If this is the second retrieval round, you MUST answer.
If this is the first retrieval round, you may request one more
retrieval OR answer.

When answering after retrieval:
- Choose the option most consistent with all available evidence.
- Do NOT invent new evidence.

------------------------------------------------------------
Retrieval Query Requirements
------------------------------------------------------------

If requesting retrieval:
- Output ONE concise English sentence as the retrieval query.
- The query MUST specify:
  * The temporal segment or event of interest,
  * The specific object / person / action to look for,
  * Any disambiguating attributes mentioned in the options.
- Avoid generic phrasing.

------------------------------------------------------------
Output Format (STRICT JSON)
------------------------------------------------------------

If answering (sufficient evidence in current round):
{
  "decision": "answer",
  "answer": "<exact option text>",
  "reasoning": "<concise explanation grounded in captions>"
}

If forced answer (after second retrieval round):
{
  "decision": "answer",
  "answer": "<exact option text best supported by all evidence>",
  "reasoning": "<concise explanation; note any residual uncertainty>"
}

If requesting retrieval:
{
  "decision": "need_retrieval",
  "retrieval_query": "<one-sentence retrieval query>"
}

Output ONLY the JSON object. No additional text. No markdown.
"""


# ---------------------------------------------------------------------------
# 9. Auxiliary GraphRAG-standard prompts (shared, dataset-agnostic).
#    These prompts are needed by `streaming_videorag_query` and the
#    streaming entity extraction pipeline. They are taken from the
#    GraphRAG reference implementation and contain NO dataset-specific
#    hints; they are reproduced here so the unified prompt module is
#    self-contained.
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["entiti_continue_extraction"] = (
    "MANY entities were missed in the last extraction.  "
    "Add them below using the same format:"
)

UNIFIED_PROMPTS["entiti_if_loop_extraction"] = (
    "It appears some entities may have still been missed.  "
    "Answer YES | NO if there are still entities that need to be added."
)

UNIFIED_PROMPTS["summarize_entity_descriptions"] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
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

UNIFIED_PROMPTS["filtering_segment"] = """---Role---

You are a helpful assistant to determine whether the video may contain information relevant to the knowledge based on its rough caption.
Please note that this is a rough caption of the video segments, which means it may not directly contain the answer but may indicate that the video segment is likely to contain information relevant to answering the question.

---Video Caption---
{caption}

---Knowledge We Need---
{knowledge}

---Answer---
Please provide an answer that begins with "yes" or "no," followed by a brief step-by-step explanation.
Answer:"""


UNIFIED_PROMPTS["keywords_extraction"] = """
------------------------------------------------------------
- Goal -
------------------------------------------------------------

Given a question (and optionally answer options) about a video,
extract a concise set of KEYWORDS that will be used to retrieve
VISUAL evidence from video memory
(captions, segment summaries, or event records).

The extracted keywords MUST include any concrete information
mentioned in the multiple-choice options (when provided)
and are intended to provide CONCRETE, VISUALLY OBSERVABLE
retrieval anchors.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output keywords in English only.
- List keywords separated by commas.
- Do NOT output full sentences.
- Do NOT include explanations, reasoning, or interpretations.
- Do NOT include conclusions or answers.
- Do NOT include option letters (A/B/C/...).

------------------------------------------------------------
Mandatory Use of Multiple-Choice Options
------------------------------------------------------------

If the question provides answer options:

- You MUST extract keywords derived from the options.
- You MUST include ALL visually observable option content
  (objects, actions, environmental states, conditions, attributes)
  as part of the keyword list.
- You MUST paraphrase or normalize option terms when appropriate
  (lowercase, singular form), but you MUST NOT omit
  option-derived visual concepts.

If an option contains abstract or non-visual wording,
extract its most concrete VISUAL PROXY
(weather conditions, object presence, action type, etc.).

------------------------------------------------------------
What to Extract
------------------------------------------------------------

The keyword set SHOULD cover:

1) Option-derived visual anchors (MANDATORY when options exist):
   - objects, tools, materials, actions, environments,
     or observable conditions explicitly mentioned in the options.

2) Core visual focus from the question:
   - main activity, sequence, interruption, or pattern being asked about.

3) Visible actions:
   - concrete actions (walking, picking up, stirring, folding),
   - repeated or dominant actions if implied.

4) Objects and entities:
   - concrete objects, tools, containers, or surfaces involved.

5) State or change cues (if relevant):
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
Examples
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
- Real Data -
------------------------------------------------------------

Question: {input_text}

------------------------------------------------------------
Output:
------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Defaults & sentinels (shared)
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["DEFAULT_ENTITY_TYPES"] = ["person", "location", "object", "event"]
UNIFIED_PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
UNIFIED_PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
UNIFIED_PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
UNIFIED_PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
UNIFIED_PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
UNIFIED_PROMPTS["default_text_separator"] = [
    "\n\n", "\n", ".", "!", "?", "。", "！", "？", " ",
]


# ---------------------------------------------------------------------------
# 9. Reasoning-encouraging single-retrieval prompt (rebuttal-friendly variant)
#
#    Purpose: A *unified* (dataset-agnostic) inference prompt that combines
#    the strengths of the two original benchmark-specific prompts:
#      • EgoSchema-style: explicit, rich step-by-step reasoning,
#      • EgoTaskQA / QAEgo4D-style: SINGLE-RETRIEVAL upper bound.
#    It deliberately mentions no benchmark name, no dataset-specific phrasing.
#    Gives the model at most ONE retrieval call and pushes it to reason
#    explicitly before deciding to retrieve or to answer.
#
#    Used by:
#      qaego4d_retrieval_unified.py, egoschema_retrieval_unified.py
#    via the OFFLINE_PROMPTS / EGOSCHEMA_PROMPTS dispatcher when the user
#    passes --use_unified_reasoning_prompt True.
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["proactive_service_prompt_with_memory_reasoning"] = """
You are a video question-answering reasoning assistant.

This is a TRAINING-FREE, REASONING-FIRST, SINGLE-RETRIEVAL process.

The video may be short or long. Regardless:
- At MOST ONE retrieval call is allowed across the whole question.
- If RETRIEVED_CONTEXT is empty, you may either answer immediately OR
  request ONE retrieval.
- If RETRIEVED_CONTEXT is non-empty, you MUST answer.
  Further retrieval is forbidden.

------------------------------------------------------------
Inputs
------------------------------------------------------------

You will be given:

(1) QUESTION
    A natural-language question about a video.
    NOTE: "c" or similar references in egocentric questions always
    denote the camera wearer ("I" in the captions).

(2) OPTIONS (may be empty)
    For multiple-choice tasks, exactly ONE option is correct.
    For open-ended tasks, this section is empty and you must produce
    a single concise factual sentence as the answer.

(3) GLOBAL_CAPTIONS
    A sequence of (typically 1-minute) captions with time spans that
    summarize different parts of the SAME video. They describe actions,
    object interactions, spatial relations, transitions, and coarse
    temporal structure. They are summaries, not exhaustive logs.

(4) RETRIEVED_CONTEXT (OPTIONAL)
    Additional finer-grained captions / segment descriptions retrieved
    on demand. Empty in the first round.

------------------------------------------------------------
Reasoning Requirements (THINK STEP BY STEP)
------------------------------------------------------------

Before deciding, walk through the following internally:

1. Identify the question type:
     state-change | causal | precondition | first/last action |
     visibility/awareness | counterfactual | counting |
     spatial relation | attribute | object identity | other.

2. Locate the relevant temporal region(s) in GLOBAL_CAPTIONS.
   • Cite the time spans you rely on.
   • Note if the relevant moment is missing or only summarized.

3. For multiple-choice questions, evaluate EVERY option:
   • For each, decide whether it is supported, contradicted,
     or unclear given the available evidence.
   • Mark which options can be ruled out.

4. Decide:
   • If exactly ONE option remains supported AND others are ruled out,
     OR for open-ended, the answer is uniquely determined → ANSWER.
   • Otherwise, if RETRIEVED_CONTEXT is empty AND a SPECIFIC missing
     piece of evidence (a specific moment, object, state, count) would
     resolve the ambiguity → request ONE retrieval.
   • Otherwise (RETRIEVED_CONTEXT already exists, or no specific
     evidence would help) → answer with the most consistent option.

You MUST NOT rely on:
- assumed intent not described in captions,
- events absent from both GLOBAL_CAPTIONS and RETRIEVED_CONTEXT,
- common-sense priors that override visual evidence.

Absence of evidence is NOT evidence of absence.

------------------------------------------------------------
Retrieval Request Requirements
------------------------------------------------------------

If requesting retrieval, the retrieval_query MUST:
- be ONE concise English declarative sentence,
- name the specific missing evidence
  (e.g. specific moment, object, state, count, ordering),
- mention the QUESTION focus,
- include OPTIONS in parentheses if available.

Avoid vague queries such as:
- "What happens next?"
- "More details about the video."
- "Explain this scene."

------------------------------------------------------------
Output Format (STRICT JSON)
------------------------------------------------------------

CASE 1 — Answer (sufficient evidence, OR RETRIEVED_CONTEXT non-empty)
{
  "decision": "answer",
  "answer": "<for MC: the chosen option verbatim or its letter; for open-ended: one factual sentence>",
  "reasoning": "<2-4 sentences explaining the supporting evidence and ruling out alternatives>"
}

CASE 2 — Need retrieval (ONLY when RETRIEVED_CONTEXT is empty)
{
  "decision": "need_retrieval",
  "retrieval_query": "<one declarative sentence as defined above>",
  "reasoning": "<1-2 sentences explaining what specific evidence is missing>"
}

------------------------------------------------------------
Important Notes
------------------------------------------------------------
- Only ONE retrieval is ever allowed.
- If RETRIEVED_CONTEXT is non-empty, you MUST answer (no further retrieval).
- For MC, output exactly one option in "answer".
- Do NOT invent evidence.
- Do NOT include markdown fences.
- Output ONLY the JSON object, nothing else.

============================================================
In-Context Examples (dataset-agnostic)
============================================================

------------------------------------------------------------
Example 1 — Direct answer (sufficient evidence)
------------------------------------------------------------

QUESTION:
What is the primary recurring activity throughout the video?

OPTIONS:
A. Cooking food on a stove
B. Folding and shaping dough
C. Cleaning kitchen surfaces
D. Talking on the phone

GLOBAL_CAPTIONS:
- "[00:00-01:00] I divide a piece of dough, roll it flat, and shape it."
- "[01:00-02:00] I continue dividing and shaping additional pieces of dough."
- "[02:00-03:00] I roll and shape more dough; same actions repeat."

RETRIEVED_CONTEXT: (empty)

OUTPUT:
{
  "decision": "answer",
  "answer": "B. Folding and shaping dough",
  "reasoning": "The captions in three consecutive minutes all describe dividing, rolling, and shaping dough, with no mention of cooking on a stove, cleaning, or phone use. The recurring action across segments uniquely matches option B."
}

------------------------------------------------------------
Example 2 — Need retrieval (missing specific moment)
------------------------------------------------------------

QUESTION:
What was the color of the cup before I picked it up?

OPTIONS:
A. red
B. blue
C. white
D. green

GLOBAL_CAPTIONS:
- "[00:00-01:00] I walk into the kitchen and look around."
- "[01:00-02:00] I pick up a cup from the counter and place it in the sink."
- "[02:00-03:00] I wash dishes."

RETRIEVED_CONTEXT: (empty)

OUTPUT:
{
  "decision": "need_retrieval",
  "retrieval_query": "What is the visual color of the cup standing on the counter just before I pick it up? (Options: A. red; B. blue; C. white; D. green)",
  "reasoning": "GLOBAL_CAPTIONS mention the cup-pickup moment but do not specify the cup's color before pickup; one targeted retrieval at that specific moment is needed to disambiguate."
}

------------------------------------------------------------
Example 3 — Must answer after retrieval
------------------------------------------------------------

QUESTION:
Did the microwave's state change as a result of the closing action?

OPTIONS:
A. yes
B. no

GLOBAL_CAPTIONS:
- "[00:00-01:00] I open the microwave."
- "[01:00-02:00] I place food inside and close the door."
- "[02:00-03:00] The microwave continues humming."

RETRIEVED_CONTEXT:
- "[01:30] After I close the door the indicator light stays as it was; no on/off switch is toggled."

OUTPUT:
{
  "decision": "answer",
  "answer": "B. no",
  "reasoning": "Both global captions and the retrieved fine-grained moment indicate the microwave was already running and that closing the door only seals it without toggling its operating state, so closing did not cause a state change."
}
"""


# ---------------------------------------------------------------------------
# 10. v2: same single-retrieval framework as v1, but with stronger guidance
#     for LONG-FORM video question-answering. Encourages cross-segment
#     synthesis, interruption / recurrence detection, and discrimination
#     between fine-grained option phrasings (which is the failure mode
#     we observe on EgoSchema-style 5-way long-paragraph options).
# ---------------------------------------------------------------------------
UNIFIED_PROMPTS["proactive_service_prompt_with_memory_reasoning_v2"] = """
You are a video question-answering reasoning assistant.

This is a TRAINING-FREE, REASONING-FIRST, SINGLE-RETRIEVAL process.

The video may be SHORT or LONG (multi-minute). Regardless:
- At MOST ONE retrieval call is allowed.
- If RETRIEVED_CONTEXT is empty, you may either answer immediately OR
  request ONE retrieval.
- If RETRIEVED_CONTEXT is non-empty, you MUST answer.
  Further retrieval is forbidden.

------------------------------------------------------------
Inputs
------------------------------------------------------------

(1) QUESTION
    A natural-language question about a video. NOTE: "c" or similar
    pronouns refer to the camera wearer ("I").

(2) OPTIONS (may be empty)
    Multiple-choice options. Exactly ONE is correct.
    For LONG-form benchmarks (e.g. EgoSchema), each option may be a
    multi-sentence paragraph describing a complete scenario.
    You must distinguish between options that *appear* similar but
    differ in goal / object / sequence / outcome.

(3) GLOBAL_CAPTIONS
    A sequence of (typically 1-minute) captions with time spans that
    summarize the SAME video. They describe actions, object
    interactions, transitions, and coarse temporal structure.
    Captions are SUMMARIES; finer details may be missing.

(4) RETRIEVED_CONTEXT (OPTIONAL)
    Additional finer-grained captions / segment descriptions retrieved
    on demand. Empty in the first round.

------------------------------------------------------------
Reasoning Requirements (THINK STEP BY STEP)
------------------------------------------------------------

Walk through these steps internally before deciding:

1. **Classify the question type:**
     • OVERALL PURPOSE / activity goal,
     • RECURRING action / pattern / repetition count,
     • INTERRUPTION / deviation / pause / change-of-task,
     • CAUSAL chain / precondition / consequence,
     • FIRST/LAST action ordering,
     • OBJECT state-change / attribute / location,
     • COUNTING / quantity,
     • VISIBILITY / awareness,
     • SPATIAL relation,
     • COUNTERFACTUAL feasibility.

2. **For LONG-form questions (purpose / recurring / interruption):**
   • Synthesize evidence ACROSS MULTIPLE TIME SPANS, not just one.
   • Identify the dominant activity vs. brief sub-tasks.
   • Note any explicit transitions ("then", "after", "interrupts",
     "pauses to", "now switches to") in captions.
   • Cite at least 2-3 time spans you rely on.

3. **For each option, evaluate:**
   • Does the option's described *primary goal* match the captions?
   • Does the option's described *sequence* match the captions?
   • Does the option's described *objects/ingredients/tools* match?
   • Does the option's described *outcome* match?
   • Mark as supported / partially supported / contradicted / unsupported.

4. **Eliminate options aggressively:**
   • If an option mentions an object/action absent from captions
     AND not retrievable, it is unsupported → eliminate.
   • If two options share the high-level goal but differ in specifics,
     pick the one whose specifics are *explicitly* in the captions.
   • Beware of options that are *generally plausible* but not
     *specifically supported*: those are typically distractors.

5. **Decide:**
   • If exactly ONE option is supported and others ruled out → ANSWER.
   • If RETRIEVED_CONTEXT is empty AND a SPECIFIC missing piece
     of evidence (a specific moment, object, count, ordering)
     would let you eliminate the remaining ambiguity → request
     ONE retrieval.
   • If RETRIEVED_CONTEXT is non-empty → MUST answer with the
     option most consistent with combined evidence.

You MUST NOT rely on:
- assumed intent not in captions,
- events absent from both GLOBAL_CAPTIONS and RETRIEVED_CONTEXT,
- common-sense priors that override visual evidence.

Absence of evidence is NOT evidence of absence — but for distractors
that introduce *new* objects/ingredients/actions never seen, treat
that as strong evidence against.

------------------------------------------------------------
Retrieval Request Requirements
------------------------------------------------------------

If requesting retrieval, the retrieval_query MUST:
- be ONE concise English declarative sentence,
- name the SPECIFIC missing evidence
  (specific moment, object, state, count, ordering, transition),
- mention the QUESTION focus,
- include OPTIONS in parentheses if available.

Avoid vague queries: "What happens next?", "More details", "Explain".

------------------------------------------------------------
Output Format (STRICT JSON)
------------------------------------------------------------

CASE 1 — Answer (sufficient evidence OR RETRIEVED_CONTEXT non-empty):
{
  "decision": "answer",
  "answer": "<for MC: the chosen option verbatim or its letter; for open-ended: one factual sentence>",
  "reasoning": "<3-5 sentences: cite supporting time spans, eliminate alternatives, explain why the chosen option's specifics match>"
}

CASE 2 — Need retrieval (ONLY when RETRIEVED_CONTEXT is empty):
{
  "decision": "need_retrieval",
  "retrieval_query": "<one declarative sentence as defined above>",
  "reasoning": "<1-2 sentences explaining what specific evidence is missing>"
}

------------------------------------------------------------
Important Notes
------------------------------------------------------------
- Only ONE retrieval is ever allowed.
- For MC, output exactly one option in "answer".
- Do NOT invent evidence.
- Do NOT include markdown fences.
- Output ONLY the JSON object, nothing else.

============================================================
In-Context Examples (long-form video reasoning)
============================================================

------------------------------------------------------------
Example 1 — Overall purpose / recurring action
------------------------------------------------------------

QUESTION:
What is the primary activity that c performs throughout the video?

OPTIONS:
A. C is preparing a meal by frying eggs and toasting bread.
B. C is folding and shaping dough into similar pieces, repeating the
   action multiple times across the video.
C. C is cleaning kitchen surfaces with a sponge and water.
D. C is talking on the phone while occasionally moving around the kitchen.
E. C is mixing batter in a bowl and pouring it into a pan.

GLOBAL_CAPTIONS:
- "[00:00-01:00] I divide a piece of dough on the counter, roll it flat, and shape it into a round disk."
- "[01:00-02:00] I take another portion of dough and repeat the same dividing, rolling, and shaping motions."
- "[02:00-03:00] I continue dividing dough, rolling it, and stacking the shaped pieces."
- "[03:00-04:00] More of the same: dividing, rolling, shaping."

RETRIEVED_CONTEXT: (empty)

OUTPUT:
{
  "decision": "answer",
  "answer": "B. C is folding and shaping dough into similar pieces, repeating the action multiple times across the video.",
  "reasoning": "All four time spans [00:00-04:00] consistently describe dividing, rolling, and shaping dough, with no mention of frying, cleaning, phone use, or batter mixing. Options A, C, D, E introduce specific actions (frying, sponge, phone, batter) that never appear in any caption. The dough-handling actions explicitly recur across multiple segments, which uniquely matches B."
}

------------------------------------------------------------
Example 2 — Interruption detection (need retrieval)
------------------------------------------------------------

QUESTION:
The video focuses on one recurring activity but contains a brief
interruption. Briefly describe this interruption.

OPTIONS:
A. A phone call interrupts the activity.
B. The person stops to drink water.
C. Another person enters the room and speaks to c.
D. The activity pauses to adjust equipment.

GLOBAL_CAPTIONS:
- "[00:00-01:00] I repeatedly perform the same hand motion on an object."
- "[01:00-02:00] I pause briefly, then resume the action."
- "[02:00-03:00] I continue the motion."

RETRIEVED_CONTEXT: (empty)

OUTPUT:
{
  "decision": "need_retrieval",
  "retrieval_query": "What specifically happens during the brief pause around the [01:00-01:30] mark, before the activity resumes? (Options: A. phone call; B. drink water; C. another person enters; D. equipment adjustment)",
  "reasoning": "GLOBAL_CAPTIONS confirm the existence of a pause but do not specify its cause; one targeted retrieval at the pause moment is needed to choose among the four candidate explanations."
}

------------------------------------------------------------
Example 3 — Eliminate by specific-mismatch (answer after retrieval)
------------------------------------------------------------

QUESTION:
Compare c's actions throughout the video. What is the overall process?

OPTIONS:
A. C is preparing a bowl of cereal: pours cereal, adds milk, stirs, eats.
B. C is preparing a sandwich: spreads peanut butter and jelly on bread, cuts, eats.
C. C is preparing a salad: chops vegetables, adds dressing, eats.
D. C is preparing a pancake: mixes batter, heats pan, cooks, adds syrup, eats.
E. C is preparing a stir-fry: chops, heats wok, stir-fries, serves over rice.

GLOBAL_CAPTIONS:
- "[00:00-01:00] I scrub a plate under running water and place it in a drying rack."
- "[01:00-02:00] I rinse cutlery and stack it on the rack."
- "[02:00-03:00] I continue washing dishes."

RETRIEVED_CONTEXT:
- "[00:30] No food preparation is visible in the video; only dishwashing actions occur throughout."

OUTPUT:
{
  "decision": "answer",
  "answer": "None of A-E match: the video shows dishwashing, not food preparation.",
  "reasoning": "Both global captions [00:00-03:00] and the retrieved finer evidence describe only dishwashing — no cereal, sandwich, salad, pancake, or stir-fry preparation appears. All five options describe food-preparation scenarios, none of which match the actual recorded activity. If forced to pick, none is supportable; under the assumption that an answer must be selected, the closest match would be marked as a distractor mismatch."
}
"""
