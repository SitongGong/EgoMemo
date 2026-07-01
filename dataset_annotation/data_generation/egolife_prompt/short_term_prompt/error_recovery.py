ERROR_RECOVERY_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (the actual span is determined by the timestamps in the input, typically ≤ 2.5 hours).

Your job in this stage is signal mining and dialogue generation for a special Short-term Proactive Service subtype:

Short-term Error-Recovery Proactive Service
(short: Short-term Error-Recovery)

This service focuses on workflow-level mistakes inside ongoing tasks with short-term window (10s-10min), where:
	•	The user is performing a multi-step task or configuration workflow.
	•	The annotations show that they have entered a wrong state, used the wrong object/target, or executed the wrong step.
	•	Correcting the situation requires the user to roll back to an earlier valid state (undo or revert) and then redo the step correctly, not just adjust technique.
	•	A timely assistant reminder can help the user fix the mistake before it propagates (wasted effort, invalid results, broken setup, etc.).

Typical examples:
	•	The user inserts a battery in the wrong orientation and needs to remove it and reinsert it correctly.
	•	The user pours a liquid into the wrong container and must pour it back or restart in the correct container.
	•	The user selects the wrong menu option or mode in a multi-step configuration flow and must undo and re-select the correct one.
	•	The user plugs a cable into the wrong port or attaches the wrong adapter and must detach it and reconnect to the intended interface.
	•	The user performs “step 3” of a procedure without having done “step 2”, and must roll back to the correct step order.

You must:
	1.	Scan the current batch to detect short-term error-recovery opportunities grounded in the annotations.
	2.	For each error, select the current moment where:
		•	the wrong state is already present,
		•	continuing from this state would propagate or lock in the mistake, and
		•	a rollback reminder would still be timely and helpful.
	3.	For each such moment, generate a short multi-turn dialogue where the assistant:
		•	points out the workflow error,
		•	suggests how to undo or roll back,
		•	and offers concrete steps to redo the action correctly.

There is no historical memory JSON for this service:
	•	You treat each batch independently.
	•	You do not track cross-day or long-horizon workflows.
	•	You only output dialogues for this batch, each grounded in one current error-recovery episode.

You must NOT guess new errors, invent timestamps, or hallucinate behaviors not supported by the annotations.

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

#### Fine-grained evidence priority

When detecting Short-term Error-Recovery opportunities, you must prioritize fine-grained entries:
	•	i_do_steps
	•	interactions_with_objects
	•	interactions_with_people
	•	speakers_say

Treat a fine-grained entry as a potential Error-Recovery episode only if its content clearly indicates a wrong step, wrong object, wrong target, wrong configuration, or invalid workflow state.

This priority applies both to:
	•	which content you treat as the error episode, and
	•	which time_window you copy for that episode.

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
	•	service_main_type: “Short-term Proactive Service”
	•	service_sub_type: “Error-Recovery Proactive Service”

Note: this service operates on short-term, local workflow mistakes that require rollback + redo, and is distinct from:
	•	Instant Safety (immediate physical injury risk),
	•	Tool Use (suboptimal technique when the workflow state is still valid),
	•	Episodic Task Reminder (unfinished tasks inside one episode),
	•	Episodic Memory Recall (short-horizon episodic memory across ≤2 hours),
	•	Habit-Coaching or Long-Term services (chronic posture or lifestyle issues),
	•	Long-Horizon Memory-Link (cross-day or ≥2h memory links).

⸻

### **What counts as an Short-term Error-Recovery signal?**

An Short-term Error-Recovery opportunity is defined by one current error episode, characterized by:

1. A concrete workflow error
The annotations indicate that the user has:
	•	chosen the wrong object or resource (wrong container, wrong tool, wrong component),
	•	executed a wrong step of a procedure,
	•	applied a wrong configuration (incorrect setting, mode, or parameter),
	•	inserted or attached something in an invalid way (wrong orientation, wrong port),
	•	or otherwise put the system into an invalid or undesired state that does not match the intended workflow.

2. Recovery requires rollback, not just technique adjustment
To proceed correctly, the user must:
	•	undo or reverse an action (e.g., remove the battery, detach the connector),
	•	move something back to an earlier valid resource (e.g., pour liquid back, move items back),
	•	revert a setting (e.g., switch mode back, uncheck a wrong option),
	•	or go back to a previous step and repeat correctly.

If the situation can be fully fixed by small technique adjustments (changing grip, angle, speed) while staying in the same workflow state, it is not Error-Recovery; it belongs to Tool Use.

3. No acute physical hazard (otherwise → Safety)
If the same wrong step also creates immediate physical risk (cut, burn, shock, collision within a few seconds), then it should be treated as Instant Safety, not Error-Recovery.

4. Not a purely missing step (otherwise → Episodic Task Reminder)
If the issue is that the user never performed a required step and is now moving away (unfinished task), but they are not in an invalid state that requires rollback, this belongs to Episodic Task Reminder, not Error-Recovery.

5. Not a long-horizon or habit issue
Long-term posture, repeated strain, or lifestyle risk without an explicit wrong step in a multi-step workflow belongs to other services (Habit-Coaching, Long-Term), not Error-Recovery.

⸻

### **Mutual Exclusion & Priority vs. Other Services**

Use these rules to distinguish Error-Recovery from other proactive services:

1. Error-Recovery vs. Tool Use  (critical distinction)
This is the most important boundary and must be enforced strictly.
	•	Tool Use:
		•	The user is performing the correct task and correct step, but with suboptimal technique.
		•	Examples: unstable grip on a screwdriver, holding a knife awkwardly, standing in a position that makes leverage poor.
		•	The workflow state is valid; the result might be less efficient or less precise, but does not require undoing previous steps.
		•	The assistant message sounds like:
			•	“Hold it this way for better control.”
			•	“If you adjust your angle, it will be easier.”
	•	Error-Recovery:
		•	The user has performed a wrong step or wrong choice, or created an invalid system state.
		•	Technique changes alone cannot fix it; the user must roll back (undo or revert) and redo correctly.
		•	The assistant message sounds like:
			•	“You might have placed this in the wrong slot; you may want to remove it and connect it to the other one.”
			•	“This liquid went into the wrong container; you could pour it back and use the labeled cup instead.”

A useful test:  If perfect technique from this point on would still leave the system in the wrong state, it is Error-Recovery, not Tool Use.

2. Error-Recovery vs. Instant Safety
	•	Instant Safety: main concern is immediate physical injury (cut, burn, slip, collision, electric shock) in the next few seconds.
	•	Error-Recovery: main concern is task correctness and valid workflow state; consequences are incorrect results, wasted effort, or broken setup, not immediate bodily harm.

If a wrong step also causes acute danger (e.g., mis-wiring near exposed live contacts), classify as Safety for this Short-term Proactive Service.

3. Error-Recovery vs. Episodic Task Reminder
	•	Episodic Task Reminder: the user has not yet completed a task or planned step within the episode and is switching away, but the current state is not necessarily invalid. A reminder is about finishing a missing step.
	•	Error-Recovery: the user has already performed a wrong step or wrong choice that must be undone; the reminder is about rolling back and correcting the workflow.

4. Error-Recovery vs. Next-Step Guidance
	•	Next-Step Guidance: the user is in a valid state and simply needs help deciding or remembering what the next correct step should be.
	•	Error-Recovery: the user is in an invalid state and must go backwards before any next step makes sense.

5. Error-Recovery vs. Habit-Coaching / Long-Term
	•	Long-term patterns (working posture, screen distance, frequency of breaks) without a specific wrong step in a concrete workflow should not be labeled Error-Recovery.

⸻


### **Time-Window Rules**

For every Short-term Error-Recovery dialogue you output, you must define a single:
	•	current_time_window representing the error episode and the moment when the assistant speaks.

Rollback Memory Horizon (critical)
When determining whether a rollback is needed and which earlier step/state it depends on, you must follow these temporal limits:
	•	The relevant earlier step or valid state must end within 10 seconds to 5 minutes before the current_time_window.
	•	In all cases, you must never use evidence older than 10 minutes when establishing the rollback chain.
	•	If the only correct reference state is older than this horizon, you must not classify the situation as Short-term Error-Recovery (it belongs to long-horizon services instead).

Rules:
	1.	Current time window (current_time_window)
		•	Prefer to copy from one fine-grained entry’s time_window:
		•	i_do_steps[].time_window, or
		•	speakers_say[].time_window.
		•	Do NOT merge or fabricate new ranges.
	2.	Instant horizon
		•	current_time_window should correspond to the short segment where:
			•	the incorrect state is present and visible in the annotations, and
			•	a rollback reminder would still be in time to prevent the error from propagating.

The assistant’s first proactive utterance is interpreted as happening during or immediately after this current_time_window.

⸻

### **Short-term Error-Recovery Detection Objective**

For the current batch (no external history):
	1.	Scan all segments and interaction_records to find candidate error episodes where:
		•	fine-grained annotations describe a wrong object, wrong step, wrong configuration, invalid connection, or invalid state.
	2.	For each candidate, verify that:
		•	There is a specific workflow error (wrong resource, wrong step, wrong orientation, wrong target).
		•	Technique adjustments alone cannot fully fix it; rollback is required.
		•	There is no immediate physical danger (otherwise → Safety).
		•	The situation is not purely a missing step where the user is walking away (otherwise → Episodic Task Reminder).
	3.	For each valid error episode:
		•	Choose the most fine-grained time_window capturing the error moment.
		•	Record the local scene context and a short explanation of what is wrong and what must be undone.
	4.	For each such episode, generate:
		•	One structured Short-term Error-Recovery entry with:
		•	current_segment_id, current_time_window, supporting_source,
		•	a neutral scene description,
		•	error_key, error_summary, rollback_required, rollback_reason,
		•	occurrence_confidence.
		•	One multi-turn dialogue where the assistant suggests rolling back and redoing correctly.

You may output multiple error_recovery_events in one batch, but:
	•	Do not duplicate events for trivially overlapping time_windows describing the same error.
	•	If multiple fine-grained annotations describe the same continuous error, you may choose one representative time_window.

⸻

### **Dialogue Generation Objective**

For each Short-term Error-Recovery episode, you must generate a short multi-turn dialogue that:
	1.	Grounds in the specific workflow error
		•	The assistant should explicitly or implicitly mention:
		•	what was done incorrectly (wrong object/step/setting/orientation),
		•	why this state is invalid for the task,
		•	and that going back one step will help.
		•	Examples:
		•	“It looks like this cable is connected to the other port; you might want to move it to the port labeled X instead.”
		•	“You poured this into the unlabeled cup; the instructions refer to the marked measuring cup. You could pour it back and use that one.”
		•	“This battery seems reversed; you may want to remove it and reinsert it with the plus side aligned to the mark.”
		•	Do not mention “video”, “annotations”, “models”, or internal logging.
	2.	Keeps focus on the current moment
		•	Make it clear that the guidance is about the error as it exists now, e.g.:
		•	“Before you continue with the next step…”
		•	“Right now, this setting doesn’t match the intended mode.”
		•	“As it’s currently connected, this won’t work as expected.”
	3.	Offers concrete rollback actions
		•	Suggest one or two simple steps to revert and redo:
		•	remove / detach / undo,
		•	move back / revert to previous container or location,
		•	change setting back to default and reselect,
		•	go back to the missing or correct step.
	4.	Tone and emotional value
		•	Calm, non-judgmental, supportive.
		•	Emphasize helping the user avoid wasted effort or confusion:
		•	“so you don’t have to redo everything later,”
		•	“to keep this setup consistent with the instructions.”
		•	Avoid scolding language (“you did it wrong again”); instead:
		•	“If you’d like, you can undo this step and try it this way…”
		•	“You might want to adjust this before moving on.”
	5.	Dialogue structure

		Each proactive_dialogue should contain at least 2 turns, preferably 3-4:
		1.	Assistant (Turn 1)
			•	Proactively identifies the error,
			•	explains that rollback will help,
			•	and suggests specific recovery actions.
		2.	User (Turn 2)
			•	Natural reply (≥ 12 English words).
			•	May:
			•	accept and undo the step,
			•	partially accept (e.g., ask for details),
			•	decline (“I intended to do it this way”),
			•	or clarify the goal.
		3.	Assistant (optional Turn 3)
			•	Acknowledge the user’s choice,
			•	provide more precise guidance, or
			•	gracefully back off if the user declines.
		4.	User (optional Turn 4)
			•	Short acknowledgment or final confirmation.

	Diversity

	Across different error-recovery dialogues in the same output:
		•	Vary how you introduce the rollback:
		•	“You might want to undo this step…”
		•	“Before you go on, it could help to reverse that…”
		•	“It may work better if you go back one step and adjust this.”
		•	Vary how you describe the benefit:
		•	“to match the intended setup,”
		•	“so the result follows the instructions,”
		•	“so you don’t need to redo more work later.”

	Not every user must comply; some may choose to proceed intentionally.

⸻

### **Output Format**

Your output MUST be ONE JSON object:
{
  "service_main_type": "Short-term Proactive Service",
  "service_sub_type": "Error-Recovery Proactive Service",

  "error_recovery_events": [
    {
      "event_id": "erecovery_001",

      "current_segment_id": "<segment id where the assistant speaks>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "scene_description": "<short neutral description of what the user is doing around this moment>",
      "trigger_reason": "<why this moment is an Error-Recovery opportunity, grounded in the annotations>",

      "error_key": "<abstract description of the workflow error, e.g., 'wrong container', 'incorrect orientation', 'wrong port', 'invalid step order'>",
      "error_summary": "<short explanation of what is wrong and why rollback is needed>",
      "rollback_required": true,
      "rollback_reason": "<why the user needs to undo and redo the step instead of only adjusting technique>",
      "risk_type": "workflow_error",
      "risk_immediacy": "immediate",
      "potential_downstream_issue": "<optional: what could happen if the user continues without fixing it (invalid result, wasted effort, etc.)>",

      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant issues rollback guidance, grounded in the error and context>"
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
	•	error_recovery_events may contain 0, 1, or multiple entries depending on how many Short-term Error-Recovery opportunities exist in this batch.
	•	potential_downstream_issue is optional and should stay consistent with the annotations (e.g., “measurement result may be invalid”, “mixture may not follow the intended recipe”).

⸻

#### Failure Case Format

If you find no Short-term Error-Recovery opportunities in the current batch:
{
  "service_main_type": "Short-term Proactive Service",
  "service_sub_type": "Error-Recovery Proactive Service",
  "error_recovery_events": [],
  "note": "No Short-term Error-Recovery Proactive Service opportunities were detected in the current batch, given the current annotations."
}
"""