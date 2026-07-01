MEMORY_RECALL_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (the actual span is determined by the timestamps in the input, typically within ≤ 2.5 hours).

Your job in this stage is signal mining and dialogue generation for a special Episodic Proactive Service subtype:

Episodic Memory Recall Proactive Service
(short: Episodic Recall Service)

This service focuses on short-horizon episodic memory within the same day and within ~2 hours, where:
	•	Something the user did/said/placed earlier in this batch (the past episode)
	•	Becomes relevant and helpful now (the current episode),
	•	And the assistant can proactively remind or resurface that past episode at the right moment.

Examples:
	•	The user put their notebook on a side table 30 minutes ago; now they are leaving the room without it.
	•	The user said "I'll send that email after finishing this call" 40 minutes ago; now they are idling at the laptop after the call.
	•	The user prepared ingredients or tools a while ago; now they are about to start the related task.

***You must:***
	1.	Scan the current batch to detect episodic links within ≤ 2 hours (no cross-day and no external history).
	2.	For each strong link, select one current moment where a reminder would be useful.
	3.	For each such moment, generate one short multi-turn dialogue where the assistant recalls the past episode and offers concrete, helpful action.

There is no historical memory JSON in this service:
	•	You treat each batch independently.
	•	You do not need to output or update any persistent memory state.
	•	You only output dialogues for the current batch, each grounded in one current episode and one past episode.

You must NOT guess new events, invent timestamps, or hallucinate behaviors not supported by the annotations.

⸻

### **Authoritative Timestamps & Fine-Grained Evidence Priority**

Each fine-grained annotation entry has a time_window aligned to the original video, for example:
	•	i_do_steps[].time_window
	•	interactions_with_objects[].time_window
	•	interactions_with_people[].time_window
	•	speakers_say[].time_window

All time_window strings already have the format:

DAY# HH:MM:SS-HH:MM:SS

You MUST only copy these existing time_window values.
Do NOT interpolate, merge, extend, shrink, or invent new time ranges.

### **Fine-grained evidence priority**
When detecting episodic memory links, you must prioritize fine-grained entries:
	•	i_do_steps
	•	interactions_with_objects
	•	interactions_with_people
	•	speakers_say

Each past episode and current episode in a recall link must correspond to at least one fine-grained entry whose content supports that interpretation.

⸻

### **Input Human Annotations**

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

These interaction_records and their subfields are your only evidence.

The coverage of this batch is defined by the minimum and maximum timestamps appearing in these JSONs.
You can assume this coverage is at most about 2.5 hours.

⸻

Service Category
	•	service_main_type: "Episodic Proactive Service"
	•	service_sub_type: "Episodic Memory Recall Proactive Service"

But note: this service operates on short-horizon episodic memory (≤ 2 hours) and does not maintain long-horizon or cross-day memory state.

⸻

### **What counts as an Episodic Memory Recall signal?**

An Episodic Recall opportunity is defined by a pair of episodes within the same batch:
	1.	A past episode (earlier in time in this batch) where the user does/says/places something that can be useful later.
	2.	A current episode (later in time, within ≤ 2 hours of the past one) where recalling that past episode would clearly help the user.

We distinguish three common recall types:
	1.	Forgotten / mis-placed item (forgotten_item)
        •	Past: the user puts or leaves an item somewhere (bag, table, shelf, other room).
        •	Current: the user:
            •	is about to leave the place without it, or
            •	is looking for / mentioning the item, or
            •	starts a task where that item is obviously needed.
        •	The reminder: "That thing you used/put earlier is over there."
	2.	Unfinished or deferred plan (unfinished_plan)
        •	Past: the user expresses an intention or plan to do something later within this session, e.g.:
            •	"After this I'll send the email / check the results / tidy up here."
            •	"I'll come back to finish labeling these boxes."
        •	Current: the user:
            •	reaches a natural moment where that plan could be executed (idle at computer, done with prior task), or
            •	mentions or starts a related task in a way that suggests they might have forgotten the earlier plan.
        •	The reminder: "You said you wanted to do X after Y, do you want to handle it now or schedule it?"
  3. Contextual cue, status check, or inspired use (context_cue / check_status / inspired_use)
        Past: The user previously started a short process, left a cue, or heard/said a tip that may soon matter (e.g., starting a download, putting food in the oven, charging a device, hearing a useful instruction or trick).
        Current: The user returns to the relevant context (oven, desk, charging spot, device) or begins a task where that earlier process/tip becomes directly helpful.
        Reminder:
            •	check_status: "You started X earlier — do you want to check its status or use it now?"
            •	context_cue: "You left/mentioned Y earlier — it might be relevant here."
            •	inspired_use: "Someone/you mentioned Z earlier — do you want to try that approach now?"

Time constraint (critical):
	•	The past episode and the current episode must be within the same batch and within a short horizon ≤ 2 hours.
	•	Anything that spans > 2 hours or different days belongs to Long-term Memory-Link service, not this one.
	•	Very short gaps (a few seconds) that only reflect micro-steps of the same immediate action should be treated as local task structure, not Episodic Recall.

Simple test:
	•	If the main value is "remember what just happened earlier in this session (≤ 2h) to avoid forgetting or to complete a plan" → candidate for Episodic Recall.
	•	If the main value is:
        •	"build a habit" → Habit-Coaching Proactive Service.
        •	"optimize a routine or default config" → Routine Optimization Proactive Service.
        •	"track long-term progress / skill improvement" → Personal Progress Feedback Proactive Service.
        •	"link events across days or long gaps (≥ 2h)" → Long-Horizon Memory-Link Proactive Service.

⸻

#### Exclusions

Do NOT classify as Episodic Recall:
	•	One-off random actions with no clear short-horizon future relevance.
	•	Generic tidying/arranging without any obvious near-future scenario tied to it.
	•	Pure routine optimization ("you always do this sequence, let me bundle it") without a specific past episode within this batch.
	•	Long-term health/productivity habits (hydration / posture / breaks).
	•	External safety issues (hot surfaces, tripping, etc.) → Safety Proactive Service.
	•	Long-horizon links (≥ 2h or cross-day) → Memory-Link Contextual Proactive Service.

⸻

### **Time-Window Rules (strict)**

You must define for every Episodic Recall dialogue:
	•	A past_time_window representing the earlier episode.
	•	A current_time_window representing the moment when the assistant speaks up.
    •	Each past_time_window MUST correspond to exactly one specific earlier event that is directly used in the current_time_window episode; the past and current events must form a clear **one-to-one correspondence** for the recall link.

Rules:
	1.	Past time window (past_time_window)
        •	MUST be copied exactly from one fine-grained entry's time_window:
            •	i_do_steps[].time_window, or
            •	interactions_with_objects[].time_window, or
            •	interactions_with_people[].time_window, or
            •	speakers_say[].time_window.
        •	Do NOT use interaction_record.time_window.
        •	Do NOT merge or fabricate new ranges.
	2.	Current time window (current_time_window)
        •	MUST be copied exactly from one:
            •	i_do_steps[].time_window, or
            •	speakers_say[].time_window.
        •	You MUST NOT use interactions_with_objects or interactions_with_people as the source for current_time_window, even if they describe related context.
        •	Choose the earliest time_window where:
            •	the current behavior first shows that the past episode is relevant (e.g., starting to leave, starting to sit at the desk, opening the app), and
            •	a reminder would still be timely and helpful.
	3.	Temporal order
        •	past_time_window MUST be strictly earlier than current_time_window.
        •	The gap between them should be more than a few seconds (so it's not the same micro-action) and at most ~2 hours.

The assistant's first proactive utterance is interpreted as happening immediately after this current_time_window.

⸻

### **Episodic Link Detection Objective**

For the current batch (no external history):
	1.	Scan all segments and interaction_records to find candidate past episodes that:
	    •	create a resource, place an object, express a near-future plan, says something, or start a short-horizon process whose status will matter soon.
	2.	For each candidate past episode, look later in the batch (within ≤ 2.5 hours) for current episodes where:
        •	the same object/plan/process becomes relevant, or
        •	the user's current behavior suggests they might forget/need that earlier episode.
	3.	For each strong pair (past episode, current episode):
        •	Confirm that both episodes are supported by specific fine-grained annotations.
        •	Confirm that the gap is within ≤ 2 hours and fits the short-horizon nature.
        •	Select one current episode as the trigger point.
	4.	For each such pair, generate:
        •	One structured recall entry with:
        •	past_segment_id, past_time_window, past_observation,
        •	current_segment_id, current_time_window, current_observation,
        •	a natural-language link_reason.
        •	One multi-turn dialogue where the assistant uses this episodic memory to help now.

You may output multiple recall dialogues in one batch, but:
	•	Never duplicate the same (past, current) pair.
	•	Do not trigger multiple times for trivially overlapping current episodes referring to the same past event.

⸻

### **Dialogue Generation Objective**

For each activated episodic recall pair, you must generate a short multi-turn dialogue that:
	1.	Grounds in the specific past episode
        •	The assistant should implicitly or explicitly reference the earlier action or statement, in a natural way, e.g.:
        •	"You left your notebook on the side table earlier."
        •	"A bit earlier you said you wanted to send that email after this call."
        •	"You started charging this device earlier this afternoon."
        •	But: do not mention "video", "annotations", "models", or internal logging.
	2.	Connects the past to the current situation
        •	Show that the reminder is context-aware, e.g.:
            •	User is leaving room: remind about item left behind.
            •	User is idle at laptop: remind about earlier plan to send email.
            •	User returns to kitchen: remind about food in oven or short process started earlier.
	3.	Offers concrete, practical help
        •	For forgotten_item:
        •	Point out the location, offer to mark it or remind again later if needed.
        •	For unfinished_plan:
        •	Suggest doing it now or scheduling it later, possibly summarizing the plan.
        •	For context_cue / check_status:
        •	Suggest checking status, verifying results, or moving to the next step.
	4.	Tone and emotional value
        •	Gentle, non-judgmental, slightly caring.
        •	Emphasize reducing mental load ("so you don't have to keep it in your head").
        •	Avoid scolding ("you forgot again"), instead:
        •	"If you'd like, I can remind you…"
        •	"Do you want to take a moment to…"
	5.	Dialogue structure
        Each proactive_dialogue must contain at least 2 turns, preferably 3-4:
        1.	Assistant (Turn 1)
            •	Proactively recalls the past episode, links it to the current situation, and offers one or two clear options.
        2.	User (Turn 2)
            •	Natural reply (≥ 12 English words).
            •	May accept, partially accept, decline, postpone, or ask for a different behavior.
        3.	Assistant (optional Turn 3)
            •	Acknowledge the choice and either:
            •	confirm the action (e.g. "Let's quickly do it now"), or
            •	confirm the rescheduling/reminder rule, or
            •	gracefully back off.
        4.	User (optional Turn 4)
            •	Short acknowledgment or final confirmation.
        5.	Diversity
        Across different recall dialogues in the same output:
            •	Vary how you phrase the episodic connection:
                •	"A bit earlier you…"
                •	"Just a while ago you…"
                •	"Earlier in this session you…"
            •	Vary the help framing:
                •	"Do you want to grab it before you go?"
                •	"Would now be a good time to finish that?"
                •	"I can remind you again later if you prefer."

Not every user must comply; some may say "not now", "remind me in a bit", etc.

⸻

### **Output Format**

Your output MUST be ONE JSON object:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Episodic Memory Recall Proactive Service",

  "recall_dialogues": [
    {
      "recall_id": "erecall_001",
      "recall_type": "forgotten_item | unfinished_plan | context_cue | check_status",

      "current_segment_id": "<segment id where the assistant speaks>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "current_supporting_source": "i_do_steps | speakers_say",
      "current_observation": "<what is happening now that makes the past episode relevant>",
      "current_local_context": "<short description of current scene / situation>",

      "past_segment_id": "<segment id of the past episode>",
      "past_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "past_supporting_source": "i_do_steps | speakers_say",
      "past_observation": "<what happened earlier that is now useful>",

      "time_gap_seconds": 0.0,
      "link_reason": "<why this past episode is helpful to recall now within ~2 hours>",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant proactive reminder, explicitly or implicitly recalling the past episode>"
        },
        {
          "role": "user",
          "utterance": "<likely user reply (>= 12 English words)>"
        }
        // optionally 1-2 more turns, alternating assistant ↔ user
      ]
    }
  ]
}

Notes:
	•	time_gap_seconds can be approximate; it is only used to reflect the short-horizon nature (e.g., 600 for 10 minutes).
	•	recall_dialogues may contain 0, 1, or multiple entries depending on how many good episodic recall opportunities exist in this batch.

⸻

#### Failure Case Format

If you find no episodic recall opportunities in the current batch (no clear past-current pairs within ≤ 2 hours that would benefit from reminder):
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Episodic Memory Recall Proactive Service",
  "recall_dialogues": [],
  "note": "No short-horizon episodic memory recall opportunities were detected in this batch, given the current annotations."
}
"""