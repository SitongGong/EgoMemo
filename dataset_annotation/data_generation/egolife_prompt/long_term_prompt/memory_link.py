MEMORY_LINK_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (less than 2.5 hours).

Your job in this stage is signal mining for a special Long-Term Proactive Service subtype:

•	service_main_type: "Long-Term Proactive Service"
•	service_sub_type: "Memory-Link Contextual Proactive Service"
(short: Memory-Link Service)

This service is about long-horizon memory across days or long gaps within a day (typically ≥ 2 hours).
This includes:
	•	cross-day links (DAY1 → DAY2/3/…)
	•	same-day links with a substantial temporal gap (typically ≥ 2 hours) between the earlier event and the current one.

In this stage, you must perform two related but distinct subtasks:
	1.	Type A - Future-impact memory hooks (to be stored as memory for later)
    Find events in the current batch that should be remembered because they are likely to matter later or create a potential link to future behavior.
    In other words:
    "If the system remembers this now, it can help with something that happens later."

    Typical examples:
        •	"I put the spare key behind this plant (so I can find it in an emergency)."
        •	"I promised to review this document tomorrow."
        •	"I left the box under the bed for later reuse."
        •	"I created a detailed engineering plan that will need to be executed step by step later."
    All such hooks from the current batch must be output under new_memory_candidates, and will be reused as historical memory in future batches.
	
  2.	Type B - Realized memory links (use past memory now + generate dialogue)
    Using the current batch together with the historical memory JSON (past Type A hooks and their follow-ups), find current events that reuse, depend on, or are made easier by some earlier event or piece of information.
    In other words:
    "This current action works better because I remember / rely on something that happened before."

    Typical examples:
        •	retrieving the spare key from behind the plant where it was hidden yesterday
        •	using the box that was intentionally stored under the bed the day before
        •	following up on a promise or prior instruction ("I'll now review the slides I promised you yesterday.")
        •	recalling something someone said earlier that now helps solve a problem
        •	e.g., "Yesterday they mentioned this trick with the router; I'll try it now to fix the connection."
    For each such realized link, you must output a realized_memory_links entry and generate a short multi-turn dialogue where the assistant proactively leverages the remembered past event to help in the current situation.

You will receive two types of input:
	1.	Current batch annotations (required): multiple segment-level human annotations for this contiguous time window.
	2.	Historical memory summary JSON (optional): all previously mined Memory-Link candidates that might matter for future (Type A) or already had some follow-ups (Type B).

Important:
	•	If the historical JSON is missing or empty, treat this as the first mining pass and assume there are no past events.
	•	In that case, you MUST set "realized_memory_links": [] in the output, because there is no earlier memory that can be linked; you only mine new Type A hooks.

You must NOT guess new events, invent timestamps, or hallucinate behaviors not supported by the annotations.

⸻

### **Authoritative Timestamps**

Every annotation entry has a time_window aligned to the original video, for example:
	•	i_do_steps[].time_window
	•	interactions_with_objects[].time_window
	•	interactions_with_people[].time_window
	•	speakers_say[].time_window

All time_window strings already have the format:

DAY# HH:MM:SS-HH:MM:SS

You MUST only copy these existing time_window values.
Do NOT interpolate or invent new time ranges.

⸻

### **Input Human Annotations**

1. Current batch: segment-level episodic annotations
You are given MULTIPLE JSON objects, each representing a <10-minute egocentric segment:
{
  "segment_id": "<identifier for this ~10-minute segment>",
  "description": "... high-level summary of what I do in this segment ...",
  "interaction_records": [
    {
      "time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "description": "... micro-episode summary ...",

      "i_do_steps": [ ... ],
      "speakers_say": [ ... ],
      "interactions_with_objects": [ ... ],
      "interactions_with_people": [ ... ],
      "environment_state": [ ... ],
      "task_transition": [ ... ],
      "confidence": 0.0-1.0
    }
  ]
}
These interaction_records and their subfields are your primary evidence.

The coverage of this batch is defined by the minimum and maximum timestamps appearing in these JSONs.

⸻

2. Historical memory summary JSON (optional memory)

This historical JSON has the same schema as the new_memory_candidates you will output in this stage.
In later runs, the system will feed your previous new_memory_candidates as memory_events in this historical JSON.
Therefore, new_memory_candidates should only contain Type A - future-impact memory hooks that are worth remembering for future batches.

You may also receive ONE historical JSON summarizing previously mined memory-relevant events (Future-impact memory hooks) and triggered realized memory links, with the following structure:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Memory Link Contextual Proactive Service",
  "memory_events": [
    {
      "event_id": "mem_001",
      "memory_key": "<abstract description of this memory-relevant pattern>",
      "memory_type": "future_hook | realized_link | mixed",
      "memory_summary": "<how this memory pattern unfolded over previous hours/days>",

      "occurrences": [
        {
          "segment_id": "<old segment id>",
          "time_window": "DAY# HH:MM:SS-HH:MM:SS",
          "supporting_source": "i_do_steps | interactions_with_objects | interactions_with_people | speakers_say",

          "observation": "<what happened>",
          "local_context": "<local scene / surroundings>",
          "historical_context": "<how it related to earlier ones>",
          "inferred_link_role": "initial_hook | followup_use | reminder | unresolved_plan | other",
          "link_target_ids": ["mem_000", "mem_005"],

          "workflow_position": "start | middle | end | standalone",
          "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
          "occurrence_confidence": 0.0
        }
      ]
    }
  ],
  
  "historical_realized_memory_links": [
    {
      "link_id": "mlink_00Y",

      "current_segment_id": "<current segment id>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "current_observation": "<what happens now that uses/depends on a past event>",
      "current_local_context": "<current scene / situation>",
      "inferred_link_role": "followup_use | reminder | check_status | other",

      "linked_past_events": [
        {
          "event_id": "mem_00A",
          "memory_key": "<copied from history>",
          "past_segment_id": "<segment_id of the past occurrence>",
          "past_time_window": "DAY# HH:MM:SS-HH:MM:SS",
          "link_reason": "<why this current event is a follow-up to that past event>"
        }
      ],

      "workflow_position": "start | middle | end | standalone",
      "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<first turn: assistant proactively referencing the remembered past event and offering help now>"
        },
        {
          "role": "user",
          "utterance": "<likely user response>"
        },
        {
          "role": "assistant",
          "utterance": "<second assistant turn, refining help based on user response>"
        }
      ]
    }
  ]
}
This historical JSON is read-only memory and is a global aggregation over all previous time ranges.
	•	Each memory_key should describe one type of "remember this to help later" pattern, not a vague category.
	•	It should roughly answer:
        "What did I do / say / place earlier that could matter later, and in what situation will it matter?"

Examples of good memory_key:
	•	"Put spare apartment key behind the big plant near the door for emergencies"
	•	"Told labmate I will review their slides the next morning"
	•	"Store the cardboard boxes under the bed for future packing"
	•	"Write Wi-Fi password on a note next to the router for visitors"

The occurrences under that memory_key are concrete historical events:
	•	sometimes the initial memory hook (e.g., "I put the key behind the plant"),
	•	sometimes a follow-up use (e.g., "I later take the key from behind the plant").

If the historical JSON is not provided or contains an empty list, treat this as no prior memory.

⸻

#### Service Category
	•	service_main_type: Long-Term Proactive Service
	•	service_sub_type: Long-Horizon Memory-Link Contextual Proactive Service

You must:
	1.	In the current batch, find:
        •	Events that should be remembered because they may help in the future (future hooks).
        •	Events that reuse or depend on earlier memory hooks (realized links).
	2.	Integrate with historical memory events when possible:
        •	Reuse existing memory_key where appropriate.
        •	Link new follow-ups to earlier event_ids, and explicitly copy the timestamps of the linked past events into the output.
        •   For each realized memory link, generate a short multi-turn dialogue between the assistant and the user grounded in these linked events.

⸻

### **Fine-Grained Evidence Priority (critical)**

When detecting candidate occurrences, you must prioritize fine-grained entries before falling back to coarse summaries:

#### Primary sources for discovering memory-relevant events  
   • i_do_steps  
   • interactions_with_objects  
   • interactions_with_people  
   • speakers_say  

→ Treat each fine-grained entry as a potential memory hook or realized link **if its content matches the Memory-Link definition.**  
→ **You must copy the `time_window` directly from these fine-grained entries only.**

This priority applies to both:  
   • which content you consider as occurrences, and  
   • which `time_window` you copy for those occurrences.

**Strict constraints on timestamps:**  
   • You must **never** use `interaction_record.time_window`.  
   • You must **never** merge, split, expand, or shorten any existing `time_window`.  
   • You must **never** create a new or "combined" `time_window` (e.g., by joining two ranges or inventing a midpoint).  
   • For every occurrence, you must copy **one existing** `time_window` string exactly as it appears in the fine-grained annotations.

⸻

### **What counts as a Memory-Link signal?**

We distinguish two roles:
	•	Type A - Future-impact memory hook:
    Something is done/said/placed now so that it can be leveraged later.
    (Current batch → future.)
	•	Type B - Realized memory link:
    A current action clearly refers back to, or makes use of, an earlier event.
    (Past → current batch.)

Only Type A - future-impact memory hooks are written into new_memory_candidates and will be carried forward as long-horizon memory for future batches.
Type B - realized memory links are reported only in realized_memory_links for this batch and are not stored as new memory hooks.

A single occurrence is enough to record a candidate, as long as it clearly fits one of these roles.

#### A. Future-impact memory hooks (in the current batch)
You are looking for events where the user creates a future resource, landmark, or commitment that only becomes useful later if remembered.

Typical patterns:
	1.	Place or configure something for later use
        •	Hiding or storing important items in specific places for later retrieval:
            •	putting a spare key behind a plant or under a mat
            •	placing a package in a specific corner "for later pickup"
            •	storing boxes under the bed for future packing
        •	Preparing tools/materials clearly intended for future tasks:
            •	pre-labeling boxes or cables for future organization
            •	pre-charging multiple devices for tomorrow's recording
	2.	Verbal commitments, instructions, or future-facing notes
        •	Making promises or plans that require remembering later:
            •	"I'll review your slides tomorrow morning."
            •	"Let's continue this discussion after lunch."
        •	Leaving instructions or explanations meant for future self/others:
            •	explaining where something is stored or how to operate something later
            •	saying "I'm putting the spare key here so I don't forget."
    3.	Creating explicit future reference points
        •	Writing down info in a place that will be used later:
            •	writing Wi-Fi password on a note near the router
            •	labeling shelves/boxes so others can find items later
        •	Setting up long-running processes that will need attention:
            •	starting an overnight download and saying when to check it
            •	starting a slow-cooking process that must be checked later

If such an event appears now (in the current batch), you should output it as a future-impact candidate even if the actual follow-up has not happened yet.

#### B. Realized memory links (current batch uses past events)
You are looking for events where the user benefits from or explicitly references something done earlier.

Typical patterns:
	1.	Retrieving or using a previously placed item
        •	Taking the spare key from behind the plant / under the mat.
        •	Pulling out the labeled box that was prepared the day before.
        •	Using the pre-charged device that was plugged in yesterday for today's task.
	2.	Following up on a prior promise, plan, or instruction
        •	Reviewing the slides that were promised for "tomorrow".
        •	Joining a meeting that was scheduled in a previous day's clip.
        •	Resuming a task exactly where a previous "pause" or bookmark was left.
	3.	Explicit verbal reference to past actions
        •	"I left this here yesterday so I could find it today."
        •	"This is the box I prepared last time."
        •	"As I said before, I'll now check the results."

Strict requirement for realized_memory_links:
You MUST only output a realized link when you can directly associate the current event with at least one concrete past occurrence from the historical JSON (or from earlier in this batch when historical JSON is present).
Each realized link must explicitly list which past events it is linked to.
Additionally, for within-day links:
    • Only treat a current event as a realized_memory_link if the linked past occurrence is separated by a clearly later episode in time (typically ≥ 2 hours apart, or obviously different sessions such as a morning planning clip vs. an evening execution clip).
    • Short within-episode references (e.g., a few minutes apart in one continuous task) should not be labeled as long-horizon realized links; those belong to local task structure or routine optimization instead.

If no suitable past event can be found in the historical JSON, do NOT create a realized link; instead treat the current event as:
	•	a new future-impact hook (if it is forward-looking), or
	•	not Memory-Link (if purely local).

⸻

#### Exclusions

Do NOT classify as Memory-Link events:
	•	one-off random actions with no clear future relevance
	•	generic organization with no clear future scenario (e.g., quick tidy-up without any hint it matters later)
	•	pure safety actions → Safety Proactive Service
	•	short-horizon corrections within the same small task → Short-Horizon Error-Recovery
    •   short-gap references within the same session (e.g., a step referring to something done a few minutes ago within one continuous workflow), even if they "mention the past", but do not require remembering across a long gap (≥ 2 hours) or a distinct episode → treat as local task structure or Routine Optimization, not Memory-Link.
	•	long-term health/productivity habits (hydration / posture / breaks) → Habit-Coaching Proactive Service
	•   Pure configuration preferences or generic multi-step efficiency routines **whose main value is "make the current workflow smoother/faster" but do NOT clearly depend on being remembered for a future moment**  
        → Routine Optimization Proactive Service (this includes both "pure configuration preferences" and "generic efficiency routines" that are not true Memory-Link cases)

Simple memory test:
	•	If the usefulness is "right now only" → probably NOT Memory-Link.
	•	If the main value is "I/others can remember this later and it will help then" → candidate for Memory-Link.

⸻

### **Memory Key Granularity Rules**

Goal: keep each memory_key narrow and concrete enough so that it corresponds to one clear type of "remember this for later" pattern, instead of a giant mix.

For every memory_key, you should be able to summarize all its occurrences with one short template:

"When I [do/say/place X] in [this kind of situation], it is so that later I / others can [do Y more easily]."

Only when all occurrences naturally fit this one sentence should they share the same memory_key.

Bad examples (too broad / vague):
	•	"General Organization"
	•	"Technical Discipline"
	•	"Remember Things"

Good examples:
	•	"Hide spare apartment key behind the plant near the door for emergencies later"
	•	"Store moving boxes under the bed so they are ready for future packing"
	•	"Tell labmate I'll review their slides the next morning"
	•	"Write Wi-Fi password on a note next to router for guests"

If you cannot write such a single "remember this for later" sentence without becoming vague, the key is too broad and should be split.

⸻

### **Historical Integration Rules**

1. When to reuse an existing memory_key
Given a new occurrence in the current batch and an existing memory_key from history, reuse that key (and its event_id) only if:
	1.	Target alignment
        It involves the same object / info / resource
        (e.g., same spare key, same specific boxes, same Wi-Fi note, same slides).
	2.	Memory role alignment
        The new occurrence fits the same type of memory pattern:
        •	same kind of hook (placing/storing/committing for later), or
        •	same kind of follow-up use / reference.
	3.	Future/Current scenario alignment
        The imagined or actual later scenario is similar:
        •	same emergency key usage,
        •	same future packing task,
        •	same "review tomorrow" pattern.

If these three align:
	•	Reuse the same memory_key and event_id in memory_events.
	•	Add a new occurrence under it.
	•	Set inferred_link_role appropriately:
        •	initial_hook for the first "store/commit for later" behavior,
        •	followup_use when the user benefits from it later,
        •	unresolved_plan if the plan is stated but not yet followed up in this batch.

2. When to create a NEW memory_key
Create a new memory_key (and a new event_id) when:
  • The object/info being stored is clearly different  
    - e.g., different physical item, different key set, different document group, different device.  
  • The future use scenario is different  
    - e.g., emergency retrieval vs. routine convenience vs. social courtesy (like "remind me to bring a gift").  
  • The memory pattern or storage strategy is structurally different  
    - e.g., "tell a person verbally" vs "leave a written note" vs "place it in a fixed drawer" as part of a different routine.

**When you create a new event_id, you MUST:**
  • Use the global `"mem_XXX"` format (e.g., `mem_001`, `mem_002`, …).  
  • Scan all existing `event_id` in the historical JSON, find the maximum numeric index, and assign the next integer (zero-padded to 3 digits).  
  • Example: if existing IDs are `mem_001` and `mem_002`, the next new `event_id` MUST be `mem_003`.  
  • You MUST NOT rename, reuse, or renumber any existing `event_id`.

⸻

### **Objective**

For the current batch of segments (time span determined by timestamps, often less than ~2.5 hours), with optional historical memory, you must perform both long-horizon memory state updating and trigger activation:
	1.	Update long-horizon memory state (new_memory_candidates / memory_events)
        • Scan all segments and interaction_records in the current batch to find every occurrence that qualifies as a future-impact memory hook (Type A), i.e., actions, placements, or verbal commitments that are clearly intended to help at some later moment if remembered.
        • When deciding whether to reuse an existing memory_key or create a new one, you MUST follow the Memory Key Granularity Rules strictly, so that each memory_key consistently represents one coherent "remember this for later" pattern.
        • When integrating new occurrences with the historical memory summary JSON, you MUST follow the Historical Integration Rules strictly, including:
            - Reusing an existing memory_key / event_id only when the target, memory role (hook vs follow-up), and imagined future scenario align with the existing pattern.
            - Creating a new memory_key and a new event_id (in the global "mem_XXX" scheme) when the object/info, future scenario, or storage strategy is meaningfully different.
        • For each memory_key (identified by event_id), append new occurrences capturing:
            - segment_id, time_window, supporting_source (from fine-grained entries only),
            - observation, local_context,
            - historical_context (how this relates to earlier hooks/uses under the same memory_key),
            - inferred_link_role (initial_hook or unresolved_plan for Type A),
            - workflow_position, social_dynamics, and occurrence_confidence.
        • For each memory_key you touch in this batch, update:
            - memory_summary: how this batch extends, confirms, or refines the long-horizon memory pattern,
            - batch_consistency_level and batch_confidence: your overall judgment of how coherent and reliable the pattern looks in this batch.

	2.	Decide current-batch realized memory links and generate dialogues (realized_memory_links)
        • Using the historical memory_events (if provided) together with all fine-grained entries in the current batch, scan for realized memory links (Type B) where a current action or utterance clearly uses, depends on, or explicitly refers back to a past memory hook, with a sufficiently long temporal gap (typically ≥ 2 hours or clearly different sessions such as "yesterday vs today" or "morning planning vs evening execution").
        • For each candidate realized link:
            - Identify the earliest fine-grained i_do_steps[].time_window or speakers_say[].time_window in the current batch where the user first starts to use the remembered object/info/plan (e.g., begins retrieving the stored item, opens the document they promised to review, starts using the pre-prepared resource).
            - Set this as current_time_window and choose supporting_source accordingly (i_do_steps or speakers_say).
            - Link this current occurrence to at least one past occurrence in the historical memory JSON by filling linked_past_events with the corresponding event_id, memory_key, past_segment_id, and past_time_window, and describe the link_reason (why the current behavior is a follow-up to that past event).
            - Set inferred_link_role appropriately (e.g., followup_use, reminder, check_status, other), and record workflow_position, social_dynamics, and occurrence_confidence.
        • For every activated realized link, you MUST also generate a short multi-turn proactive dialogue under proactive_dialogue that:
            - Is initiated by the assistant at current_time_window,
            - Explicitly and helpfully leverages the remembered past event to assist the user now (e.g., retrieving the plan they made, checking steps they prepared, reminding of where they stored something),
            - Follows the dialogue rules defined later in this prompt (assistant starts, user responds naturally, assistant follows up with a concrete micro-service),
            - Stays grounded in the linked past event and the current physical/interaction context, without mentioning videos, annotations, or models.
        • If no historical JSON is provided or it is empty:
            - You may still output new_memory_candidates from this batch (Type A hooks),
            - But you MUST set "realized_memory_links": [] because there is no earlier memory to link to.
        • The "historical_realized_memory_links" field in the Historical Memory Summary JSON is used to assist in activating memory-link events.
⸻

### **Trigger Time-Window Rules (strict)**

When you output **current_time_window** for any trigger, you MUST:

1. **Only** copy an existing `time_window` string from:
   - `i_do_steps[].time_window`, or  
   - `speakers_say[].time_window`.

2. You may still use `interactions_with_objects` and `interactions_with_people` as **supporting evidence** for detecting habits or patterns,  
   but you are **NOT allowed** to use their `time_window` fields directly as `current_time_window`.

3. You must copy one of the allowed time_window strings **exactly as-is**.  
   Do **NOT** invent, merge, extend, or shrink any time range.

If in doubt, keep separate occurrences with their own `current_time_window` instead of fusing them.

⸻

### **Output Format (single JSON with two sections)**

Your output must be ONE JSON object that contains both types of results for the current batch:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Memory Link Contextual Proactive Service",

  "new_memory_candidates": [ <Type A future-impact memory hook>
    {
      "event_id": "mem_00X",
      "memory_key": "<abstract but concrete memory pattern>",
      "memory_summary": "<summary of this memory pattern within the current batch, optionally referencing earlier days>",
      "memory_type": "future_hook",

      "occurrences": [
        {
          "segment_id": "<source segment file name>",
          "time_window": "DAY# HH:MM:SS-HH:MM:SS",
          "supporting_source": "i_do_steps | interactions_with_objects | interactions_with_people | speakers_say",

          "observation": "<specific action or speech that creates a future-impact memory hook>",
          "local_context": "<what is around / scene state> (optional but recommended)",
          "historical_context": "<how this occurrence relates to previous hooks of the same memory_key, if any>",
          "inferred_link_role": "initial_hook | unresolved_plan",
          "link_target_ids": [],

          "workflow_position": "start | middle | end | standalone",
          "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
          "occurrence_confidence": 0.0
        }
      ],

      "batch_consistency_level": "high | medium | low",
      "batch_confidence": 0.0
    }
  ],

  "realized_memory_links": [
    {
      "link_id": "mlink_00Y",

      "current_segment_id": "<current segment id>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "current_observation": "<what happens now that uses/depends on a past event>",
      "current_local_context": "<current scene / situation>",
      "inferred_link_role": "followup_use | reminder | check_status | other",

      "linked_past_events": [
        {
          "event_id": "mem_00A",
          "memory_key": "<copied from history>",
          "past_segment_id": "<segment_id of the past occurrence>",
          "past_time_window": "DAY# HH:MM:SS-HH:MM:SS",
          "link_reason": "<why this current event is a follow-up to that past event>"
        }
      ],

      "workflow_position": "start | middle | end | standalone",
      "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<first turn: assistant proactively referencing the remembered past event and offering help now>"
        },
        {
          "role": "user",
          "utterance": "<likely user response>"
        },
        {
          "role": "assistant",
          "utterance": "<second assistant turn, refining help based on user response>"
        }
      ]
    }
  ]
}
Hard constraints for realized_memory_links:
	•	If no historical JSON is provided, you MUST set:
        "realized_memory_links": []
	•	For every realized_memory_links item:
        •	linked_past_events MUST contain at least one entry, each with event_id, past_segment_id, and past_time_window copied from the historical JSON.
        •	proactive_dialogue MUST be a multi-turn dialogue (≥ 3 turns) showing how the assistant could use the remembered past event to help the user in the current situation.
        •	The dialogue must sound like a natural real-time assistant (do NOT mention "video", "annotations", or "model").
  •	***Do not repeatedly activate the same event or generate identical dialogue content.***

*** Additional Dialogue Rules: ***
    - For every realized_memory_links item, current_time_window MUST be exactly the time range where the user is first seen speaking/acting in a way that uses or starts to use the past event (e.g., the moment they begin the move, open the document, start the dance practice), not a later summary window.
    - For realized_memory_links, current_time_window MUST be copied from either i_do_steps[].time_window or speakers_say[].time_window only. You MUST NOT use time windows from interactions_with_objects or interactions_with_people as current_time_window, even if they describe related context.
    - If multiple i_do_steps / speakers_say entries are involved in the same realized link, you MUST choose the earliest time_window where the user first starts the memory-dependent action or verbal reference (the first step or first utterance that actually uses the past event).
    - The assistant must initiate help right at this current_time_window, when the remembered information becomes practically useful in the ongoing action.
    - The first assistant utterance should offer a concrete, task-relevant assistance based on the linked past event (e.g., "Do you want me to follow the moving plan you made yesterday?").
    - The user's first reply should either confirm or refine the assistance; the assistant must then follow up with a helpful action (checking, summarizing, retrieving info, validating progress).
    - The dialogue must stay grounded in the current physical action and the past memory link, without referencing videos, annotations, or models.
    - The final assistant turn should complete a useful micro-service (e.g., verifying steps, retrieving past notes, confirming plans, or tracking progress) based strictly on the remembered information.⸻

If no historical JSON is provided:
	•	You can still output new_memory_candidates for hooks detected in this batch.
    •	In subsequent batches, the new_memory_candidates you output here will be loaded back as memory_events in the historical JSON. Treat them as seeds of long-horizon memory that future realized links may refer to.
	•	For every new memory_key:
        •	memory_summary: "initial hypothesis of a possible long-horizon memory pattern in this batch."
        •	historical_context in its occurrences: "no prior records; first detected in this batch."

⸻

#### Failure Case Format

If you find no memory-link related occurrences in the current batch:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Memory Link Contextual Proactive Service",
  "new_memory_candidates": [],
  "realized_memory_links": [],
  "note": "No long-horizon memory-related hooks or realized links were detected in this batch, given the current annotations and historical summary."
}
"""