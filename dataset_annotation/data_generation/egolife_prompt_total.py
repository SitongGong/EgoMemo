from __future__ import annotations
from typing import Any
from egolife_prompt.instant.safety import SAFETY_PROMPT
from egolife_prompt.instant.tool_use import TOOL_USE_PROMPT
from egolife_prompt.episodic_prompt.memory_recall import MEMORY_RECALL_PROMPT
from egolife_prompt.episodic_prompt.task_reminder import TASK_REMINDER_PROMPT
from egolife_prompt.short_term_prompt.error_recovery import ERROR_RECOVERY_PROMPT
from egolife_prompt.short_term_prompt.next_step_guidance import NEXT_STEP_GUIDANCE_PROMPT
from egolife_prompt.short_term_prompt.resource_reminder import RESOURCE_REMINDER_PROMPT
from egolife_prompt.long_term_prompt.habit_coaching import HABIT_COACHING_PROMPT
from egolife_prompt.long_term_prompt.memory_link import MEMORY_LINK_PROMPT
from egolife_prompt.long_term_prompt.personal_feedback import PERSONAL_FEEDBACK_PROMPT
from egolife_prompt.long_term_prompt.routine_optimization import ROUTINE_OPTIMIZATION_PROMPT


PROMPTS: dict[str, Any] = {}

PROMPTS["egolife_summarize_system_prompt"] = """
### **Task Context**

You are analyzing one ~10-minute **egocentric annotation JSON dictionary** recorded from the first-person perspective (the camera wearer = "I"). This segment is extracted from a multi-day continuous recording in the EgoLife dataset.

All information you can use comes from the input **human annotation JSON**.

### Authoritative timestamps

The JSON contains time-stamped annotations that correspond to the on-frame overlay in the original video.

- Each annotation entry has `start_time` and `end_time` (e.g., `"11:09:43"`), representing wall-clock time within a specific `DAY#`.
- Treat these times as the **only authoritative time source**.
- The single `DAY#` can be inferred from the json dictionary key (e.g., `DAY1` for `A1_JAKE_DAY1_11090000`).
Use this same `DAY#` prefix for all `time_window` fields you output.
- Every `time_window` must align **exactly** to these annotation boundaries.
Format: `DAY# HH:MM:SS-HH:MM:SS`.
Do **not** guess or interpolate new times.

### **Input JSON Dictionary**

You are given exactly one JSON object for a continuous ~10-minute segment. Its structure is roughly:

- `dense_caption`: a list of fine-grained action annotations about me ("我 …"), each with
    - `index`: integer order
    - `start_time`, `end_time`: `"HH:MM:SS"`
    - `text`: short natural-language description of what *I* am doing (often in Chinese)
- `transcript`: a list of dialogue annotations, each with
    - `start_time`, `end_time`: `"HH:MM:SS"`
    - `speaker`: string (e.g., `"I"`, `"Person A"`, `"Person B"`)
    - `utterance`: what that person says
- `description`: a natural-language summary describing what I am doing and how I interact with objects / environment / people across the entire segment.

Dense captions focus on **my actions** over short intervals.

The transcript covers **all spoken dialogue** in this ~10-minute segment.

### **Authority & Fusion Policy**

- The **annotation JSON is the primary source of truth**.
    - `dense_caption` describes what *I* am doing.
    - `transcript` describes what people say.
    - `description` is a high-level, natural-language summary of what *I* do **and how I interact with the environment and other people** across the entire segment, integrating actions, objects, locations, and major transitions.
- You may reasonably infer simple scene labels (e.g., "kitchen", "office") and coarse environment states from the captions if they are obvious, but **do not invent complex actions or intentions** beyond the annotations.
- If there is any ambiguity or conflict, prefer the **literal meaning of the annotations**.
- You must **center the camera wearer**:
    - All records should be about **what I do, what I see, what I interact with**, and how others interact with me.
- Filter out annotations that describe *only other people* and do not affect me (no interaction, no address, no shared object).

### **What to Produce**

You must **consolidate and reorganize** the annotation JSON into a **structured, factual, first-person JSON** that will serve as long-term episodic memory.

Your JSON has:

1. A **top-level description** summarizing everything about **me** in this ~10-minute window.
2. A list of **interaction_records**, each representing a coherent micro-episode (a small contiguous interval where my activity is stable).
    - Each record has its own `time_window` (non-overlapping, sorted by time).
    - Within each record, you will list: my atomic actions, relevant speech, object interactions, people interactions, environment states, and task transitions.
    - For each interaction_record, include a local `"description"` field:
        - It must be a concise, factual, first-person summary (2-4 sentences).
        - It should capture:
            - what I am mainly doing in this micro-episode,
            - the relevant objects/people involved,
            - any meaningful transitions within the record,
            - changes in scene or focus.
        - It must not include future or past episodes outside this time_window.

You should **merge** adjacent annotations into the same interaction_record when they are part of one continuous micro-task (gap ≤ 2s and same context), and **start a new interaction_record** when my activity clearly changes (e.g., from using my phone at the table to walking to my bedroom, from giving a stand-up meeting to cleaning up boxes, etc.).

### **What to Record (per interaction_record)**

For each interaction_record, fill the following fields:

- **i_do_steps**: all atomic actions I perform in this record.
    - Derived mainly from `dense_caption` entries describing *my* actions.
    - Each step must keep its own precise `time_window`.  
      **Do not merge multiple steps together even if they are related.**
    - Example:
        - “I pick up the cup” → one entry  
        - “I walk to the counter” → another entry  
        - “I place the cup down” → another entry  
    - You must **not** collapse several actions into a long combined step like “I handle the cup”.
    - Only when two dense_caption entries clearly describe the **same atomic action** *and the gap is ≤2 seconds*  
      may you merge them into a minimal covering `time_window`; otherwise keep them separate.
- **speakers_say**:
    - All dialogue lines in this record from the `transcript`.
    - Include both my own speech (`speaker: "I"`) and others' speech (`speaker: "<Name>"`).
    - Use utterances **exactly** as in the transcript (no paraphrasing).
- **interactions_with_objects** (**fine-grained requirement**):
    - Objects I touch, manipulate, use, hand over, receive, or point to.
    - Use the dense captions (and simple inference from them) to identify object names whenever possible (e.g., "phone", "cup", "laptop", "box").
    - **Treat each distinct interaction phase as a separate entry.**
        - If my action verb or interaction type changes, you MUST create a new entry with its own `time_window`.
            - Example: for a phone,  "hold while looking" → "pass to others" must be three separate entries, not one big window.
        - Do **not** merge "pick_up", "hold", "place", "pass", etc. into one coarse description.
    - You may only merge adjacent entries for the **same object AND same interaction type** when:
        - they are clearly a smooth continuation,
        - and the gap between their time_windows is ≤ 2 seconds.
- **interactions_with_people** (**fine-grained requirement**):
    - Interactions that **involve me**, such as:
        - conversation (back-and-forth dialogue, or someone speaking directly to me),
        - hand_over (I pass/receive an object),
        - gaze (I look at someone or someone looks at me in a way that affects my action),
        - assistance (I help someone, or someone helps me),
        - physical_contact (touching, bumping, guiding).
    - Each entry must correspond to a **single, coherent interaction type** within a specific time_window.
        - If the interaction type changes (e.g., from gaze to conversation, from conversation to hand_over), start a new entry with its own window.
        - Do **not** compress an entire long conversation plus object exchange into one generic "interaction".
    - Each has an exact `time_window`.
- **environment_state**:
    - Coarse but useful context: location, lighting, important devices’ state, notable changes.
    - Use short phrases like `"dining table"`, `"office"`, `"bedroom"`, `"indoors near window"`.
    - Only include states that help interpret my actions (e.g., "we are in the dining area around a table with several items on it").
- **task_transition**:
    - High-level changes in my activity inside this record (or between neighboring records), described in first person.
    - Example: `"from": "working at the desk on a laptop", "to": "standing up and walking toward the kitchen"`.
- **confidence**:
    - A float from `0.0` to `1.0` reflecting how confident you are that:
        - the actions and interactions are correct, and
        - important details in this interval have not been missed.

### **Critical Rules**

- Use **first-person phrasing** for my behavior:
    - e.g., `"I pick up the phone"`, `"I walk toward my bedroom"`, `"I point at the screen"`.
- Do **NOT** hallucinate actions, objects, or dialogue not supported by the annotations.
- Every `time_window` must strictly follow this format:
    - `DAY# HH:MM:SS-HH:MM:SS`
    - where `HH:MM:SS` comes from the annotation entries.
- Keep all interaction_records in **chronological order** and avoid overlapping time_windows.
- You may ignore tiny gaps (≤ 2s) if nothing new happens, but never extend a window beyond actual annotated boundaries.
- For `interactions_with_objects` and `interactions_with_people`:
    - Default to **fine-grained** segmentation:
        - new entry whenever the action type, relation_to_me, or interaction type changes,
        - or whenever there is a meaningful pause or shift in focus.
    - Only merge when:
        - same object/person,
        - same interaction type,
        - same semantic action,
        - and gap ≤ 2 seconds.

---

### **Output Format**

Output **only** a dictionary with this structure:

{
"description": "<comprehensive, factual, first-person summary of everything I do and what happens around me in this ~10-minute segment>",
"interaction_records": [
{
"time_window": "DAY# HH:MM:SS-HH:MM:SS",

  "description": "<2-4 sentences summarizing what I do, who/what I interact with, and the micro-context within this record>",

  "i_do_steps": [
    {
      "time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "action": "<concise first-person action phrase>",
      "role": "do | participate | observe",
      "targets": ["<object/person names if applicable>"]
    }
  ],

  "speakers_say": [
    {
      "speaker": "I | <Other Person Name>",
      "utterance": "<verbatim utterance from transcript>",
      "time_window": "DAY# HH:MM:SS-HH:MM:SS"
    }
  ],

  "interactions_with_objects": [
    {
      "object": "<object name>",
      "action": "<manipulation description for this specific phase (e.g., 'picked up', 'held while looking', 'placed back on surface')>",
      "time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "relation_to_me": "manipulate | place | pick_up | adjust | point_to | show | receive | hand_over | inspect | hold"
    }
  ],

  "interactions_with_people": [
    {
      "with": "<Person Name>",
      "type": "conversation | hand_over | gaze | assistance | physical_contact",
      "topic": "<short phrase about what this interaction is about>",
      "time_window": "DAY# HH:MM:SS-HH:MM:SS"
    }
  ],

  "environment_state": [
    {
      "location": "kitchen | living room | office | dining table | hallway | bedroom | outdoors | other",
      "lighting": "natural | artificial | dim | bright | mixed",
      "devices_state": "<states of relevant visible devices (e.g., 'screens on', 'items on table', 'boxes open')>",
      "notable_changes": "<anything in the scene that changes and matters for understanding my actions>",
      "time_window": "DAY# HH:MM:SS-HH:MM:SS"
    }
  ],

  "task_transition": [
    {
      "from": "<prior micro-task in first person>",
      "to": "<next micro-task in first person>",
      "time_window": "DAY# HH:MM:SS-HH:MM:SS"
    }
  ],

  "confidence": 0.0
}

]
}

### **Filtering Guidance (first-person focus)**

**Keep**:

- "I pick up / move / hold / put down X"
- "I walk from A to B"
- "I look at someone"
- "I talk to someone" or "someone addresses me"
- "I hand an item to someone" or "someone hands me an item"

**Drop or ignore as standalone actions**:

- actions of others that do **not** involve me (no shared object, no address, no visible effect on my behavior)
- background chatter not related to me.

**Granularity rules**:

- When an object stays in the scene but my interaction with it changes, split into multiple entries, for example:
    - hold phone while looking at screen → separate entry
    - place phone back on table → separate entry
    - pass phone to others → separate entry
- Do not collapse these into a single long interaction like "I handle the phone".

**Merging small intervals**:

- If several dense_caption entries describe a *smooth continuation of the same behavior* with identical action type and targets, and gaps ≤ 2 seconds, you may merge them into one `i_do_step` or one interaction entry, using the minimal covering `time_window`.

Your final answer must be valid JSON and follow the schema above.

### **Output Example (reference only) **

{
"description": "In this 10-minute segment, I perform a sequence of simple actions at an indoor location. I handle an item on a surface, look around occasionally, and briefly exchange a few words with another person. The environment remains stable with no major changes.",

"interaction_records": [
{
"time_window": "DAY1 11:09:40-11:09:50",

  "description": "During this interval, I focus on a small object placed on a surface. I pick it up, examine it closely, and briefly respond to a question from another person nearby.",
  "i_do_steps": [
    {
      "time_window": "DAY1 11:09:40-11:09:43",
      "action": "I look down at an object resting on the surface",
      "role": "observe",
      "targets": ["object", "surface"]
    },
    {
      "time_window": "DAY1 11:09:43-11:09:47",
      "action": "I pick up the object with one hand",
      "role": "do",
      "targets": ["object"]
    },
    {
      "time_window": "DAY1 11:09:47-11:09:50",
      "action": "I examine the object closely",
      "role": "do",
      "targets": ["object"]
    }
  ],
  "speakers_say": [
    {
      "speaker": "Person A",
      "utterance": "Is that the item you're checking?",
      "time_window": "DAY1 11:09:44-11:09:46"
    },
    {
      "speaker": "I",
      "utterance": "Yes, I'm looking at it now.",
      "time_window": "DAY1 11:09:47-11:09:49"
    }
  ],

  "interactions_with_objects": [
    {
      "object": "object",
      "action": "picked up from the surface",
      "time_window": "DAY1 11:09:43-11:09:47",
      "relation_to_me": "pick_up"
    },
    {
      "object": "object",
      "action": "visually examined",
      "time_window": "DAY1 11:09:47-11:09:50",
      "relation_to_me": "inspect"
    }
  ],
  "interactions_with_people": [
    {
      "with": "Person A",
      "type": "conversation",
      "topic": "clarifying what I am examining",
      "time_window": "DAY1 11:09:44-11:09:49"
    }
  ],

  "environment_state": [
    {
      "location": "indoor area",
      "lighting": "artificial",
      "devices_state": "stable scene with items placed on a surface",
      "notable_changes": "my attention shifts between the object and the nearby person",
      "time_window": "DAY1 11:09:40-11:09:50"
    }
  ],

  "task_transition": [
    {
      "from": "looking at the object on the surface",
      "to": "picking it up and examining it",
      "time_window": "DAY1 11:09:43-11:09:47"
    }
  ],
  "confidence": 0.95
}
]
}
"""

PROMPTS["egolife_summarize_user_prompt"] = """
### ** Input JSON Dictionary (Human Annotation) **
{annotation_data}
"""

PROMPTS["long_term"] = {
"memory_link_contextual":MEMORY_LINK_PROMPT,
"habit_coaching":HABIT_COACHING_PROMPT, 
"personal_progressive": PERSONAL_FEEDBACK_PROMPT,
"routine_optimization": ROUTINE_OPTIMIZATION_PROMPT}

PROMPTS["episodic"]= {
"task_reminder":TASK_REMINDER_PROMPT,
"memory_recall":MEMORY_RECALL_PROMPT}

PROMPTS["short_term"]= {
"next_step_guidance":NEXT_STEP_GUIDANCE_PROMPT, 
"error_recovery": ERROR_RECOVERY_PROMPT,
"resource_reminder": RESOURCE_REMINDER_PROMPT}

PROMPTS["instant"]= {
"safety":SAFETY_PROMPT,
"tool_use":TOOL_USE_PROMPT}
