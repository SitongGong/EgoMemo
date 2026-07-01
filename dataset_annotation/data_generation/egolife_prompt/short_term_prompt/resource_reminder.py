RESOURCE_REMINDER_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (the batch span is determined by timestamps in the input, typically ≤ 2.5 hours).

Your job in this stage is signal mining and proactive dialogue generation for:

Short-Term Resource Reminder Proactive Service
(short: Resource Reminder)

This service focuses on short-horizon (≈10 seconds-10 minutes) mismanaged resources or incomplete closure tasks within the current episode (≤2.5 hours), where:
	•	The user has likely forgotten to close, shut off, secure, lock, or finalize a resource or state.
	•	The risk is not immediately physically dangerous (otherwise it belongs to Safety).
	•	The reminder helps the user avoid loss, damage, or unnecessary rework within the same short episode.
	•	The model must identify “closure failures” that the user is about to leave behind, based only on short-horizon evidence.

Typical examples:
	•	Forgetting to turn off a stove burner (low flame, no overflow → not yet Safety).
	•	Forgetting to turn off electrical power or water source.
	•	Leaving a document unsaved before walking away from a computer.
	•	Leaving a door not fully closed.
	•	Leaving a bottle cap loosely attached.
	•	Throwing trash away but forgetting to close the trash-can lid.

You must:
	1.	Scan the current batch to detect resource mismanagement or incomplete closure tasks grounded in the annotations.
	2.	For each case, identify the current moment where:
        •	the resource/state is left unresolved, and
        •	the user is transitioning away or about to shift context (within the short-horizon window).
	3.	Generate a short multi-turn dialogue where the assistant:
        •	proactively reminds the user of the unresolved task,
        •	explains why it matters right now,
        •	encourages a quick fix that avoids loss/damage/rework.

There is no long-term memory JSON for this service:
	•	You treat each batch independently.
	•	You do not consider cross-day or long-horizon tasks.
	•	You only output reminders for this batch, each tied to one current unresolved resource state.

The assistant must NOT hallucinate new states, objects, hazards, timestamps, or intentions.

⸻

### **Authoritative Timestamps & Fine-Grained Evidence Priority**

Each fine-grained annotation entry has a time_window aligned to the original video, for example:
	•	i_do_steps[].time_window
	•	interactions_with_objects[].time_window
	•	interactions_with_people[].time_window
	•	speakers_say[].time_window

All time_window strings already have the format:

DAY# HH:MM:SS-HH:MM:SS

When you output current_time_window, you MUST copy it only from i_do_steps[].time_window or speakers_say[].time_window.
You must NOT use interactions_with_objects, interactions_with_people, or interaction_records as the source of current_time_window.
Do NOT interpolate, merge, extend, shrink, or invent new time ranges.

Fine-grained evidence priority
	•	Use fine-grained subfields (i_do_steps, interactions_with_objects, interactions_with_people, speakers_say) as the primary sources to detect potential short-term resource issues and to select current_time_window.
	•	Only fall back to interaction_records.description and interaction_records.time_window when necessary (no fine-grained entry clearly reflects the unresolved resource or transition-away moment).

⸻

### **Input Human Annotations**

You are given multiple JSON annotation files:
{
  "segment_id": "<segment id>",
  "description": "... high-level summary ...",
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

These are your only evidence sources for detection.

The coverage of this batch is defined by the minimum and maximum timestamps appearing in these JSONs.
You can assume this coverage is at most about 2.5 hours.

⸻

Service Category
	•	service_main_type: "Short-term Proactive Service"
	•	service_sub_type: "Short-Term Resource Reminder Proactive Service"

This service is distinct from:
	•	Next-Step Guidance (which helps continue a workflow),
	•	Error-Recovery (rollback to previous step),
	•	Tool Use (micro-action correction),
	•	Safety (immediate bodily danger),
	•	Long-Term Habit Coaching (chronic patterns),
	•	Long-Horizon Memory-Link (≥2 hours or cross-day).

⸻

### **What counts as a Short-Term Resource Reminder signal?**

A resource reminder opportunity arises when ALL of the following hold:
	1.	There exists an unresolved resource / closure state
        Examples:
            •	A stove burner is still on.
            •	Water faucet left running or dripping.
            •	Computer document unsaved while user moves away.
            •	Trash-can lid open after throwing trash.
            •	Bottle cap loose after a sealing attempt.
            •	Power switch left on after using a tool.
            •	A door left ajar.

	2.	User is transitioning away
        Signals include:
            •	User moves to another room.
            •	User starts a different task.
            •	User stops interacting with the resource.
            •	User’s attention shifts to another device, person, or object.

	3.	The situation does not meet Safety criteria

        •	Flame is low and stable (not boiling over).
        •	Wet floor is not involved.
        •	Electrical equipment is off or not in a hazardous state.
        •	No imminent physical harm expected in the next few seconds.

        If a serious hazard is imminent → classify as Safety, not Resource Reminder.
	4.	Reminder is still useful now

        •	User has not walked too far away yet.
        •	The resource can still be quickly fixed.
        •	The relevant earlier interaction with the resource and the current transition-away moment both fall within the short-horizon memory window (see below).

The assistant’s proactive intervention must feel timely and grounded.

⸻

### **Mutual Exclusion vs. Other Services**

Use these distinctions carefully:

Resource Reminder vs. Safety
	•	Resource Reminder: open/unfinished resource state, not dangerous yet.
	•	Safety: the leftover state is already causing immediate bodily danger.

Resource Reminder vs. Next-Step Guidance
	•	Next-Step Guidance: user finished a correct step and needs the next logical action.
	•	Resource Reminder: user forgot a closure step after finishing a workflow or interaction with a resource.

Resource Reminder vs. Error-Recovery
	•	Error-Recovery: user must rollback due to a wrong step (invalid state).
	•	Resource Reminder: workflow is complete or valid; user just forgot to close or finalize something.

Resource Reminder vs. Tool Use
	•	Tool Use: correcting grip/angle/stability for efficiency.
	•	Resource Reminder: ensuring resource closure, not usage mechanics.

⸻

### **Time-Window Rules (strict)**

Each event MUST have:
	•	current_time_window = the timestamp of the fine-grained entry that best captures the moment the user leaves the resource unresolved and starts to transition away, within the allowed short-horizon memory window.

Rules:
	1.	Source of current_time_window
        •	Prefer timestamps from:
            •	i_do_steps[].time_window
            •	speakers_say[].time_window
        •	Do NOT merge or fabricate new ranges.
	2.	Short-Horizon Memory Window (critical)
        Short-Term Resource Reminder operates strictly within a short-horizon memory range, reflecting short-term forgetfulness inside one continuous local episode. The model must follow these temporal constraints:
            •	Immediate window (≈10 seconds)
                If the user finishes interacting with a resource (e.g., stove, faucet, power switch, bottle cap, door latch) and within ≈10 seconds begins to leave or shift tasks, treat this as a strong Resource Reminder opportunity.
            •	Typical short-horizon span (≈10 seconds - 5 minutes)
                The model may link an earlier resource-related action to the user’s current transition-away behavior only if both belong to the same continuous episode and fall within this short window (≤ 5 minutes).
            •	Maximum allowed memory span: 10 minutes
            •	You must never use any resource-related behavior older than 10 minutes as evidence for this service.
            •	If the unresolved resource state originates beyond ≈10 minutes or from a previous episode, it is not eligible for Short-Term Resource Reminder (ignore it for this service).
            •	No long-horizon or cross-episode reasoning
        Do NOT retrieve information from previous activities, previous rooms, earlier times of day, or different sessions.
        The reminder must rely exclusively on short-horizon evidence within the current local episode.

    The assistant’s first utterance is interpreted as occurring during or immediately after this current_time_window.

⸻

### **Resource Reminder Detection Objective**

For each batch:
	1.	Scan all segments and interaction_records to find candidate unresolved resources or closure tasks, where fine-grained annotations indicate that a resource was used/changed but not properly closed, turned off, or secured.
	2.	For each candidate, verify that:
        •	The closure step is missing (e.g., burner not turned off, file not saved, door not fully closed).
        •	The user begins transitioning away or shifting attention to something else.
        •	The state is non-hazardous (else → Safety).
        •	The earlier interaction with the resource and the transition-away moment are within the allowed 10 seconds-10 minutes short-horizon window (ideally 10s-5min, never >10min).
	3.	Select the most fine-grained time_window indicating when the user abandoned or left behind the closure task as current_time_window.
	4.	For each valid case, prepare:
        •	scene_description (what the user was doing around that moment),
        •	trigger_reason (why this is a short-term resource reminder),
        •	resource_key & resource_summary,
        •	potential_consequence,
        •	occurrence_confidence.

You may output multiple reminders, but avoid duplicates describing the same unresolved resource within overlapping time_windows.

⸻

### **Dialogue Generation Objective**

For each event, generate a short, natural multi-turn dialogue:

Assistant (Turn 1)
	•	Proactively reminds the user of the unresolved resource.
	•	References the specific resource (stove, faucet, file, door, etc.).
	•	Sounds timely and supportive, grounded in the current short-horizon situation.

Examples:
	•	“Just a reminder — the stove burner is still on.”
	•	“Looks like the trash-can lid is still open.”
	•	“I noticed your document isn’t saved yet.”

User (Turn 2, ≥ 12 English words)
	•	Acknowledges or explains.
	•	May return to close the resource or briefly justify their choice.

Optional Turn 3-4
	•	Assistant gives brief reassurance or a simple suggestion (“You can turn it off now to avoid wasting gas.”).
	•	User confirms closing the resource or clarifies their decision.

Tone
	•	Calm, friendly, non-judgmental.
	•	No scolding or blaming.
	•	Emphasize prevention and convenience (e.g., “so you don’t lose your work,” “to avoid wasting water”).

Do not mention “video”, “annotations”, “models”, or any internal logging.

⸻

### **Output Format**

Your output MUST be exactly one JSON object:
{
  "service_main_type": "Short-term Proactive Service",
  "service_sub_type": "Short-Term Resource Reminder Proactive Service",

  "resource_events": [
    {
      "event_id": "resource_001",

      "current_segment_id": "<segment id>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "scene_description": "<what the user was doing before or while walking away>",
      "trigger_reason": "<why this is a short-term resource reminder>",

      "resource_key": "<e.g., stove_left_on, water_running, file_unsaved, door_ajar>",
      "resource_summary": "<short explanation of what closure step was missed>",
      "potential_consequence": "<waste, mess, data loss, inconvenience>",

      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant proactive reminder>"
        },
        {
          "role": "user",
          "utterance": "<>=12-word natural reply>"
        }
        // optional turns 3-4
      ]
    }
  ]
}

Notes:
	•	resource_events may contain 0, 1, or multiple entries depending on how many Short-Term Resource Reminder opportunities exist in this batch.
	•	potential_consequence is optional but should stay consistent with the annotations (e.g., “wasting gas,” “water may overflow,” “risk of losing unsaved work”).

⸻

#### Failure Case

If no Resource Reminder opportunities appear:
{
  "service_main_type": "Short-term Proactive Service",
  "service_sub_type": "Short-Term Resource Reminder Proactive Service",
  "resource_events": [],
  "note": "No Short-Term Resource Reminder opportunities were detected in this batch."
}
"""