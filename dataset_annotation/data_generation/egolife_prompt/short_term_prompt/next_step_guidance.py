NEXT_STEP_GUIDANCE_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (typically ≤ 2.5 hours).

Your job in this stage is signal mining and dialogue generation for a special Short-term Proactive Service subtype:

Next-Step Guidance Proactive Service
(short: Next-Step Guidance)

This service is triggered after the user completes a correct step, in a workflow where the next step is:
	•	logically required,
	•	sequential,
	•	and immediately actionable within a short-horizon window (≈10 seconds-10 minutes).

The assistant gives the user the next logical step to continue the multistep workflow smoothly.

Typical examples:
	•	The user finishes chopping vegetables → assistant suggests “you can heat the pan now.”
	•	The user completes reagent preparation → assistant suggests the next experiment step.
	•	The user finishes assembling a part → assistant reminds them of the next continuation step.

This service is not about correcting mistakes, preventing danger, or adjusting technique.
It is simply about what comes next, assuming the current step is correct and valid.

⸻

### **Authoritative Timestamps & Fine-Grained Evidence Priority**

Each fine-grained annotation entry has a time_window aligned to the original video, e.g.:
	•	i_do_steps[].time_window
	•	speakers_say[].time_window
	•	interactions_with_objects[].time_window
	•	interactions_with_people[].time_window

All time_window strings already have the format:

DAY# HH:MM:SS-HH:MM:SS

When you output current_time_window, you MUST:
	•	copy it only from:
	•	i_do_steps[].time_window, or
	•	speakers_say[].time_window.

You must NOT use interactions_with_objects, interactions_with_people, or interaction_records as the source of current_time_window (though they can still be used as content evidence for detecting steps and workflows).

Do NOT interpolate, merge, extend, shrink, or invent new time ranges.

Fine-grained evidence priority

When detecting Next-Step Guidance opportunities, you must prioritize fine-grained entries:
	•	Primary sources for detecting steps and workflow structure:
	•	i_do_steps
	•	interactions_with_objects
	•	interactions_with_people
	•	speakers_say

Treat a fine-grained entry (or short sequence of entries) as a potential Next-Step episode only if its content clearly indicates:
	•	a correctly completed step in a multi-step workflow, and
	•	a natural “what’s next?” moment (pause, idle, or transition).

This priority applies both to:
	•	which content you treat as the completed-step episode, and
	•	which time_window you copy (from i_do_steps or speakers_say) for that episode as current_time_window.

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
	•	service_main_type: "Short-term Proactive Service"
	•	service_sub_type: "Next-Step Guidance Proactive Service"

This service is distinct from:
	•	Error-Recovery (wrong step → rollback),
	•	Tool Use (correct step but poor technique),
	•	Instant Safety (physical hazard),
	•	Episodic Task Reminder (unfinished tasks across an episode),
	•	Habit-Coaching / Long-Term,
	•	Long-Horizon Memory-Link (≥ 2h or cross-day).

⸻

### **What counts as a Next-Step Guidance signal?**

A Next-Step Guidance opportunity must satisfy all of the following:

⸻

1. The user has just completed a correct step
Evidence from fine-grained annotations shows:
	•	correct object usage,
	•	correct manipulation,
	•	correct preparation step,
	•	correct mode selection,
	•	correct assembly action.

The step is valid, successful, and brings the workflow forward.

⸻

2. The task is clearly a multistep sequential workflow
Examples:
	•	cooking / preparation sequences,
	•	DIY / assembly processes,
	•	experiment procedures,
	•	device setup flows,
	•	multi-step maintenance tasks.

The workflow must have forward continuity: a natural “what’s next?” moment.

⸻

3. The user is in a state that suggests readiness for the next step
For example:
	•	the user pauses after a correct step,
	•	looks around for what to do next,
	•	the workflow obviously continues but the user is idle,
	•	or the last step’s objective is clearly achieved (e.g., lid tightened, item cleaned, button pressed).

This service triggers when user intent likely requires continuation, not rollback or abandonment.

⸻

### **What does NOT count as Next-Step Guidance?**

1. Wrong step was performed → Error-Recovery
If the user:
	•	picked the wrong object,
	•	performed the wrong step,
	•	created an invalid configuration,

then the assistant must instruct rollback (Error-Recovery), not next step.

⸻

2. Technique is poor → Tool Use
If the user:
	•	holds a tool at a poor angle,
	•	performs a valid step but inefficiently,

this is Instant Tool Use, not Next-Step Guidance.

⸻

3. Immediate danger appears → Instant Safety
If a risky configuration arises (burn, cut, slip, shock risk):

→ classify as Instant Safety, not Next-Step Guidance.

⸻

4. User is abandoning an unfinished task → Episodic Task Reminder
Next-Step Guidance is used only when:
	•	the user is still within the workflow,
	•	and should proceed forward.

If the user walks away with an unfinished step → Episodic Task Reminder, not Next-Step Guidance.

⸻

### **Mutual Exclusion & Priority**

This section explains how to distinguish Next-Step Guidance from adjacent types.

Next-Step vs. Error-Recovery
	•	Next-Step:
        •	previous step is correct,
        •	workflow state is valid,
        •	next action naturally follows,
        •	no rollback required.
	•	Error-Recovery:
        •	previous step is incorrect OR created an invalid state,
        •	workflow requires rollback to repair,
        •	next logical step would fail if executed now.

Test:

“If the user proceeds without undoing anything, will the workflow still be valid?”
If NO → Error-Recovery.

⸻

Next-Step vs. Tool Use
	•	Tool Use:
        •	correct step is in progress,
        •	technique is suboptimal,
        •	assistant says “how to do”.
	•	Next-Step:
        •	step is complete,
        •	next step is “what to do next”.

⸻

Next-Step vs. Instant Safety
	•	If danger in the next 0-10 seconds → Safety.
	•	If safe and user is waiting/transitioning → Next-Step Guidance.

⸻

### **Time-Window Rules**

For every Next-Step Guidance dialogue you output, you must define a single:
	•	current_time_window representing the short interval where the user has just completed a correct step and is ready for the next one, and the moment when the assistant speaks.

Rules:
	1.	Source of current_time_window
        •	You MUST copy current_time_window only from:
            •	i_do_steps[].time_window, or
            •	speakers_say[].time_window.
        •	You must NOT use:
            •	interactions_with_objects[].time_window,
            •	interactions_with_people[].time_window,
            •	interaction_records[].time_window
        as the source for current_time_window (though they may still inform your understanding of the workflow).
        •	Never merge, fabricate, extend, or shrink time ranges.
	2.	Short-term next-step micro-interval
        •	A Next-Step moment corresponds to the short interval (≈0-10 seconds) immediately after the user finishes a correct step and is paused, waiting, or naturally transitioning.
        •	Choose the time_window that best captures when the step is completed and the assistant’s suggestion would be timely.

⸻

### **Short-Horizon Memory Window**

Next-Step Guidance operates strictly within a short-horizon memory range, reflecting short-term reasoning inside one continuous local workflow. The model must follow these temporal constraints:
	•	Immediate window (≈10 seconds)
If the user finishes a correct step and within ≈10 seconds pauses or appears to look for what to do next, treat this as a strong Next-Step Guidance opportunity.
	•	Typical short-horizon span (≈10 seconds - 5 minutes)
You may link the earlier step completion to the current pause/transition only if both belong to the same continuous workflow and fall within this short span (≤ 5 minutes).
	•	Maximum allowed memory span: 10 minutes
	•	You must not use any step completion older than 10 minutes as evidence for Next-Step Guidance.
	•	If the relevant earlier step happened more than ≈10 minutes ago or in a previous episode, it is not eligible for this service.
	•	No long-horizon or cross-episode reasoning
Do not retrieve step information from previous activities, earlier times of day, or different sessions.
Next-Step Guidance must rely exclusively on short-horizon evidence within the current local episode.

⸻

### **Next-Step Guidance Detection Objective**

For the current batch:
	1.	Scan all fine-grained annotations for completed correct steps in clearly sequential workflows.
	2.	For each candidate, verify that:
        •	The step was correctly executed (no error requiring rollback).
        •	The workflow is multistep and sequential.
        •	The user is in a pause / wait / transition state suggesting readiness for the next step.
        •	There is no concurrent safety risk dominating the situation.
        •	There is no incorrect step that must be fixed first (no Error-Recovery).
        •	No tool-usage problem dominates (i.e., technique is not the main issue).
        •	The earlier step completion and the current pause/transition lie within the 10 seconds-10 minutes short-horizon window (ideally 10s-5min).
	3.	Choose the best fine-grained time_window (from i_do_steps or speakers_say) for the moment just after step completion as current_time_window.
	4.	For each valid Next-Step episode, construct:
        •	scene_description (neutral description of what step the user just completed),
        •	trigger_reason (why this moment qualifies as Next-Step Guidance),
        •	next_step_key,
        •	next_step_summary,
        •	occurrence_confidence.
	5.	Generate a short multi-turn dialogue grounded in this context.

        ### **Dialogue Generation Objective**

        Each Next-Step Guidance episode requires a short dialogue:

        Assistant (Turn 1)
            •	Proactively proposes the next logical step,
            •	connects it explicitly to the step just completed,
            •	gives simple, practical guidance.

        Examples:
            •	“Since you’ve just finished chopping the vegetables, you can start heating the pan now.”
            •	“You’ve prepared the reagent; next you can pour it into the labeled tube.”
            •	“You finished attaching that part; now the next screw goes into the top-right hole.”

        Avoid mentioning videos, annotations, or datasets.

        ⸻

        User (Turn 2)

        A natural response (≥ 12 English words), such as:
            •	acknowledging the suggestion,
            •	asking how to proceed in more detail,
            •	confirming intention,
            •	or briefly explaining their plan.

        ⸻

        Assistant (optional Turn 3)

        Provides additional clarification, step-by-step instruction, or encouragement, or gracefully backs off if the user declines.

        ⸻

        User (optional Turn 4)

        Final acknowledgment or confirmation.

        Tone should be:
            •	calm,
            •	supportive,
            •	non-overbearing,
            •	focused on smooth task continuation.

⸻

### **Output Format**

Your output MUST be exactly one JSON object:

{
  "service_main_type": "Short-term Proactive Service",
  "service_sub_type": "Next-Step Guidance Proactive Service",

  "next_step_events": [
    {
      "event_id": "nextstep_001",

      "current_segment_id": "<segment id>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "scene_description": "<neutral description of what step the user just completed>",
      "trigger_reason": "<why this moment qualifies as Next-Step Guidance>",

      "next_step_key": "<short tag: 'heat_pan', 'tighten_next_screw', 'select_mode', etc.>",
      "next_step_summary": "<the next action the user can take>",
      "workflow_continuity": "sequential",
      "risk_immediacy": "none",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant gives next step>"
        },
        {
          "role": "user",
          "utterance": "<user reply (>= 12 words)>"
        }
        // optional further turns
      ]
    }
  ]
}

Notes:
	•	next_step_events may contain 0, 1, or multiple entries depending on how many Next-Step Guidance opportunities exist in this batch.
	•	supporting_source MUST be either "i_do_steps" or "speakers_say", consistent with the allowed sources for current_time_window.

⸻

#### Failure Case Output

If no Next-Step Guidance opportunities are detected:
{
  "service_main_type": "Short-term Proactive Service",
  "service_sub_type": "Next-Step Guidance Proactive Service",
  "next_step_events": [],
  "note": "No Next-Step Guidance opportunities were detected in this batch, given the current annotations."
}
"""