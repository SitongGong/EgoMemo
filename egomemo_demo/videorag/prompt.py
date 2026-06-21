
"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}


PROMPTS[
    "entity_extraction"
] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
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

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["default_text_separator"] = [
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


PROMPTS[
    "naive_rag_response"
] = """---Role---

You are a helpful assistant responding to a query with retrieved knowledge.

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the input data tables, suitable for the specified response length and format.
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{content_data}

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the input data tables appropriate for the response length and format.
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Notice---
Please add sections and commentary as appropriate for the length and format if necessary. Format the response in Markdown.
"""


PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """-Goal-
For a given query, generate a declarative sentence to serve as a query for retrieving relevant knowledge.

######################
-Examples-
######################

Question: What are the main characters? \n(A) Alice\n(B) Bob\n(C) Charlie\n(D) Dana
################
Output:
The main characters. (Maybe Alice, Bob, Charlie or Dana)

Question: What locations are shown in the video?
################
Output:
The locations shown in the video.

Question: Which animals appear in the wildlife footage? \n(A) Lions\n(B) Elephants\n(C) Zebras
################
Output:
The animals that appear in the wildlife footage. (Maybe Lions, Elephants or Zebras)

#############################
-Real Data-
######################
Question: {input_text}
######################
Output:
"""


PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """-Goal-
Given a question that may include scene-related information, generate a declarative sentence to serve as a query for retrieving relevant video segments.

######################
-Examples-
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
The protagonist encounters an animal in the forest.

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
A city chase scene where the main character is pursued by a car.

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
The opening scene of the film featuring specific weather conditions. (Maybe Sunny, Rainy, Snowy or Windy)

######################
-Real Data-
######################
Question: {input_text}
######################
Output:
"""



PROMPTS[
    "keywords_extraction"
] = """- Goal -
Given a query, extract the relevant keywords that can help answer the query. Please list the keywords separated by commas.

######################
- Examples -
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
animal, protagonist, forest, scene

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
color, car, chases, main character, city

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
weather, opening scene, film, Sunny, Rainy, Snowy, Windy

#############################
- Real Data -
######################
Question: {input_text}
######################
Output:
"""



PROMPTS[
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


PROMPTS[
    "videorag_response"
] = """---Role---

You are a helpful assistant responding to a query with retrieved knowledge.

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the retrieved text chunks and the information retrieved from videos, suitable for the specified response length and format.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Retrieved Information From Videos---

{video_data}

---Retrieved Text Chunks---

{chunk_data}

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the retrieved text chunks and the information retrieved from videos, suitable for the specified response length and format.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
Reference relevant video segments within the answers, specifying the video name and start & end timestamps. Use the following reference format:

---Example of Reference---

In one segment, the film highlights the devastating effects of deforestation on wildlife habitats [1]. Another part illustrates successful conservation efforts that have helped endangered species recover [2].

#### Reference:
[1] video_name_1, 05:30, 08:00  
[2] video_name_2, 25:00, 28:00 

---Notice---
Please add sections and commentary as appropriate for the length and format if necessary. Format the response in Markdown.
"""

PROMPTS[
    "videorag_response_wo_reference"
] = """---Role---

You are a helpful assistant responding to a query with retrieved knowledge.

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the retrieved text chunks and the information retrieved from videos, suitable for the specified response length and format.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Retrieved Information From Videos---

{video_data}

---Retrieved Text Chunks---

{chunk_data}

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the retrieved text chunks and the information retrieved from videos, suitable for the specified response length and format.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Notice---
Please add sections and commentary as appropriate for the length and format if necessary. Format the response in Markdown.
"""

PROMPTS[
    "videorag_response_for_multiple_choice_question"
] = """---Role---

You are a helpful assistant responding to a multiple-choice question with retrieved knowledge.

---Goal---

Generate a concise response that addresses the user's question by summarizing relevant information derived from the retrieved text and video content. Ensure the response aligns with the specified format and length.
Please note that there is only one choice is correct.

---Target response length and format---

{response_type}

---Retrieved Information From Videos---

{video_data}

---Retrieved Text Chunks---

{chunk_data}

---Goal---

Generate a concise response that addresses the user's question by summarizing relevant information derived from the retrieved text and video content. Ensure the response aligns with the specified format and length.
Please note that there is only one choice is correct.

---Notice---
Please provide your answer in JSON format as follows:
{{
    "Answer": "The label of the answer, like A/B/C/D or 1/2/3/4 or others, depending on the given query"
    "Explanation": "Provide explanations for your choice. Use sections and commentary as needed to ensure clarity and depth. Format the response in Markdown."
}}
Key points:
1. Ensure that the "Answer" reflects the correct label format.
2. Structure the "Explanation" for clarity, using Markdown for any necessary formatting.
"""


"""
------------------------------------------------------------
- Goal (EgoSchema, Multiple-Choice) -
------------------------------------------------------------

Given a first-person (egocentric) multiple-choice question from EgoSchema,
rewrite it as ONE concise declarative sentence
that can be used as a retrieval query over VISUAL EMBEDDINGS
of egocentric video segments (e.g., 30-second clips or sampled frames).

The rewritten sentence should describe
WHAT visual evidence would help distinguish
between the given answer options.

The output is NOT an answer.
It is a search query describing expected visual content.

------------------------------------------------------------
Core Principle (EgoSchema-Oriented)
------------------------------------------------------------

EgoSchema questions are open-ended but evaluated via multiple-choice options.
They typically ask about:

- the dominant or repeated activity in the video,
- the sequence or process formed by multiple actions,
- the purpose or role of object interaction (from visible use),
- interruptions or standout moments,
- overall behavior patterns or scene structure.

You are translating the question into a
VISUAL EVIDENCE QUERY that helps retrieve
the most discriminative video segment(s)
needed to choose among the options.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I") when referring to the camera wearer.
- Do NOT include explanations, reasoning, or multiple sentences.
- Do NOT mention “options” explicitly unless required below.

------------------------------------------------------------
Visual Grounding Requirements
------------------------------------------------------------

The rewritten query MUST focus on concrete, observable cues, including:
- my visible actions or movements (walking, picking up, placing, stirring, folding),
- interactions with objects or tools (using, holding, adjusting, moving),
- object states or state changes (open/closed, held/placed, closer/farther),
- spatial or environmental context (kitchen, grocery store, outdoors, workbench),
- visible sequences, repetitions, pauses, or transitions.

You MUST NOT:
- infer intent, goals, or correctness,
- describe abstract themes not grounded in action,
- name identities of other people,
- include non-visual concepts.

------------------------------------------------------------
Handling Process, Sequence, and Pattern Questions
------------------------------------------------------------

If the question asks about:
- a main process,
- a sequence of actions,
- a repeated or dominant activity,
- an interruption or notable moment,

rewrite the query to describe:
- the visible action flow,
- repeated manipulation or movement,
- the moment where the action changes, pauses, or diverges,

as something that could be visually identified
from the video frames.

------------------------------------------------------------
Temporal Information
------------------------------------------------------------

- Include time-related wording ONLY if explicitly asked
  (e.g., "earlier", "before", "during", "throughout").
- Do NOT invent temporal qualifiers.

------------------------------------------------------------
Multiple-Choice Disambiguation (IMPORTANT)
------------------------------------------------------------

If the question's options correspond to
distinct, visually distinguishable situations,
you SHOULD include them as alternatives using:

"(Maybe A, B, or C)"

ONLY when doing so helps retrieval focus on
discriminative visual evidence.

------------------------------------------------------------
Forbidden Content
------------------------------------------------------------

You MUST NOT:
- answer the question,
- include proactive service language,
- include advice or correction,
- include abstract judgments or conclusions.

------------------------------------------------------------
Examples (EgoSchema-Aligned)
------------------------------------------------------------
Question:
"Although the video is predominantly focused on one recurring action, there is an interruption in c's activity.
Which of the following best describes this interruption?
Options: A. c stops the action to interact with another object; B. c pauses briefly and then resumes the same action; C. c changes location and starts a different activity; D. c stops entirely and leaves the scene"

Output:
A segment where my recurring action pauses or changes, possibly involving interaction with another object, a brief stop, a location change, or starting a different visible activity.
------------------------------------------------------------
Question:
"What are the main ingredients and tools used during the video?
Options: A. Peas, water, salt, knife; B. Peas, water, salt, fork; C. Peas, water, salt, measuring cup, pan, spoon; D. Peas, water, salt, plate; E. Peas, water, salt, bowl"

Output:
Segments showing me handling peas, water, and salt, along with visible use of cooking tools such as a measuring cup, pan, spoon, knife, fork, plate, or bowl.
------------------------------------------------------------
Question:
"Based on the events of the video, how would you describe c's behavior in the grocery store and its purpose?"

Output:
Segments showing me moving through a grocery store, selecting items, interacting with shelves or counters, and looking around.
------------------------------------------------------------
Question:
"Which moments can be considered most significant in determining c's purpose for interacting with clothes and the hamper?"

Output:
Segments where I pick up, move, sort, or place clothes near a hamper.
------------------------------------------------------------
Question:
"What is the primary objective of interacting with the bicycle pedal and the tools used?"

Output:
A segment where I handle and adjust a bicycle pedal using tools.
------------------------------------------------------------
- Real Data -
------------------------------------------------------------

Question: {input_text}

------------------------------------------------------------
Output:
------------------------------------------------------------
"""


"""
------------------------------------------------------------
- Goal (EgoSchema) -
------------------------------------------------------------

Given a first-person (egocentric) user question about a video,
rewrite it as ONE concise declarative sentence
that can be used as a retrieval query over an egocentric memory system
(e.g., short captions, multi-scale summaries, or an event-centric knowledge graph).

The rewritten sentence should describe
WHAT observable action, interaction, object state,
or action pattern should be found in memory.

The output is NOT an answer.
It is a retrieval-oriented description of visual task evidence.

------------------------------------------------------------
Core Principle (EgoSchema)
------------------------------------------------------------

This prompt supports open-ended and multiple-choice
video understanding in EgoSchema / ESTP.

The rewritten query should focus on:
- observable actions or interactions I performed,
- objects or tools involved,
- object states or state changes,
- sequences, repetitions, or interruptions of actions,
- spatial or environmental context when relevant.

Do NOT assume:
- a predefined task or procedure,
- correctness or completion of actions,
- instructional intent.

------------------------------------------------------------
Rules (STRICT)
------------------------------------------------------------

- Output EXACTLY ONE English declarative sentence.
- Use first-person perspective ("I").
- Do NOT ask a question.
- Do NOT include explanations, reasoning, or commentary.
- Do NOT include advice, reminders, or service language.

------------------------------------------------------------
What to Focus On
------------------------------------------------------------

The rewritten sentence SHOULD express one or more of the following,
if implied by the question:

- a concrete action or interaction
  (e.g., picking up, placing, stirring, folding, walking),
- an object or tool being handled or observed,
- a visible object state or state change
  (held/placed, open/closed, closer/farther, in/out of view),
- a sequence, repeated action, or dominant activity,
- a brief interruption, pause, or transition.

If the question implies recurrence or dominance,
describe the recurring or dominant visible action or interaction,
NOT abstract habits or mental states.

------------------------------------------------------------
Temporal Information
------------------------------------------------------------

- Include temporal wording ONLY if explicitly implied
  (e.g., "earlier", "before", "during", "throughout the video").
- Do NOT invent time ranges.

------------------------------------------------------------
Multiple-Choice Questions
------------------------------------------------------------

If the question provides answer options,
rewrite the sentence to include them as visual alternatives
using the format:

(Maybe A, B, C, ...)

------------------------------------------------------------
Forbidden Content
------------------------------------------------------------

You MUST NOT:
- answer the question,
- infer intent, purpose, or success,
- include abstract themes or judgments,
- rely on non-visual knowledge.

------------------------------------------------------------
Examples (EgoSchema-Aligned)
------------------------------------------------------------

Question:
"Although the video is predominantly focused on one recurring action,
there is an interruption in c's activity.
(Options: A. c interacts with a different object; B. c pauses briefly and resumes the same action; C. c changes location and starts another activity; D. c stops and leaves the scene")

Output:
An event where my recurring action pauses or changes, involving a different object, a brief stop, a location change, or leaving the scene. 

------------------------------------------------------------

Question:
"What are the main ingredients and tools used during the video?
Options: A. peas, water, salt, knife; B. peas, water, salt, fork; C. peas, water, salt, measuring cup, pan, spoon; D. peas, water, salt, plate; E. peas, water, salt, bowl"

Output:
Events where I handle peas, water, and salt, along with visible use of cooking tools such as a knife, fork, measuring cup, pan, spoon, plate, or bowl. 

------------------------------------------------------------

Question:
"What is the primary sequence of actions performed throughout the video?"

Output:
A sequence of related physical actions I repeatedly perform, involving the same objects or tools across the video.

------------------------------------------------------------
- Real Data -
------------------------------------------------------------

Question: {input_text}

------------------------------------------------------------
Output:
------------------------------------------------------------
"""