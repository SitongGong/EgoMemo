ROUTINE_OPTIMIZATION_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (less than 2.5 hours).

Your job in this stage is signal mining for a special Long-Term Proactive Service subtype:

	•	service_main_type: "Long-Term Proactive Service"
	•	service_sub_type: "Routine Optimization Proactive Service"

You handle both:
	1.	Multi-step routines (time-structured or task-structured):
		•	e.g., afternoon cleaning routine, pre-meeting preparation, after-dinner cleanup, post-recording wrap-up.
	2.	Configuration-only patterns (repeated "usual setup"):
		•	e.g., typical AC temperature before sleep, usual lamp brightness, preferred phone mode in meetings, typical work-desk layout.
	3.	Expected-but-missing routines within a usual time window:
		•	e.g., the user cleaned the room around 3-4 PM on several days, but today at similar time they have not done it yet.

as long as they are recurring across sessions/days and suitable for long-term optimization (defaults, shortcuts, bundled flows, or timely reminders).

⸻

### **Core Pattern Types**

Within this batch, you focus on two complementary dimensions:
	1.	Pattern State (routine_state_updates)
	Concrete episodes that show the user:
		•	Running through a similar multi-step routine in a recognizable context, and/or
		•	Applying a stable configuration in a recurring situation.
	Examples:
		•	Afternoon cleaning routine with similar sequence of steps.
		•	Pre-meeting prep in a similar order (open slides → check link → test mic).
		•	Before sleep, setting AC, lights, and phone in a similar combination.
		•	Before online calls, always putting phone on vibrate and adjusting laptop volume.
	2.	Optimization Triggers (optimization_triggers + dialogues)
	Specific episodes in the current batch where, given previous occurrences of the same pattern, it is valuable for the assistant to:
		•	Bundle or streamline multiple repeated steps (routine-level optimization), or
		•	Save / reuse a configuration as a default / shortcut, or
		•	Prepare tools/environment/settings in advance for future runs, or
		•	Gently remind the user about a usually stable routine that has not yet occurred in its typical time window, and offer to:
			•	run through a quick checklist,
			•	reschedule or skip intentionally,
			•	or set a different reminder rule.

These Type B moments must be grounded in longitudinal evidence:
	•	Cross-day (DAY1 → DAY2/3/…), or
	•	Same-day but clearly separate sessions (≥ 2 hours apart or obviously distinct blocks, like "morning prep" vs "evening prep").

You must NOT guess new events, invent timestamps, or hallucinate behaviors not supported by the annotations.

⸻

### **Authoritative Timestamps & Fine-Grained Evidence Priority**

Every fine-grained annotation entry has a time_window aligned to the original video, for example:
	•	i_do_steps[].time_window
	•	interactions_with_objects[].time_window
	•	interactions_with_people[].time_window
	•	speakers_say[].time_window

All time_window strings already have the format:

DAY# HH:MM:SS-HH:MM:SS

You MUST only copy these existing time_window values.
Do NOT interpolate, merge, extend, shrink, or invent new time ranges.

Fine-Grained priority for pattern occurrences
When detecting candidate occurrences (for routine_state_updates):
•	Primary sources:
	•	i_do_steps
	•	interactions_with_objects
	•	interactions_with_people
	•	speakers_say

Treat each fine-grained entry (or a small, coherent cluster in the same context) as a potential routine or configuration episode if it clearly involves:
	•	A recognizable recurring sequence of steps, and/or
	•	A repeated configuration being applied in a familiar situation.

Timestamp rules for occurrences:
	•	For every occurrence in routine_state_updates, you must copy one existing time_window directly from:
	•	i_do_steps, or
	•	speakers_say.
	•	You must never use interaction_record.time_window for occurrences.
	•	You must never merge, split, or invent time_window ranges.

Additional strict rule for dialogue triggers
For optimization_triggers.current_time_window, you may only copy time_window from:
	•	i_do_steps[].time_window or
	•	speakers_say[].time_window.

You MUST NOT use time windows from interactions_with_objects or interactions_with_people as the source for current_time_window, even if they describe related context.

For missed routine reminders, you still must anchor current_time_window to some actual i_do_steps or speakers_say entry in the current batch that occurs in the relevant time-of-day window, representing the moment when the assistant decides to proactively check in (e.g., when the user is doing something else during the usual cleaning time).

⸻

### **Input Human Annotations**

1. Current batch: segment-level episodic annotations
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

2. Historical routine summary JSON (optional memory)
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Routine Optimization Proactive Service",
  "routine_events": [
    {
      "routine_id": "rtopt_001",
      "routine_key": "<abstract description of one pattern>",
      "routine_type": "configuration_preference | time_structured_routine | task_structured_routine",
      "routine_summary": "<how this pattern has unfolded over previous hours/days>",
      "batch_consistency_level": "high | medium | low",
      "batch_confidence": 0.0,

      "occurrences": [
        {
          "segment_id": "<old segment id>",
          "day_id": 1,
          "time_window": "DAY1 HH:MM:SS-HH:MM:SS",
          "supporting_source": "i_do_steps | speakers_say",

          "observation": "<what happened>",
          "local_context": "<local scene / surroundings>",
          "historical_context": "<how it related to earlier ones>",
          "inferred_motivation": "<behavior-grounded explanation>",
          "workflow_position": "start | middle | end | standalone",
          "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
          "implicit_avoidance": "<optional>",
          "routine_evolution": "<optional>",
          "occurrence_confidence": 0.0
        }
      ]
    }
  ],
  "historical_optimization_triggers": [
    {
      "trigger_id": "ropt_001",
      "routine_id": "rtopt_001",
      "routine_key": "<copied from routine_state_updates>",
      "routine_type": "configuration_preference | time_structured_routine | task_structured_routine",
      "trigger_type": "routine_block_optimization | configuration_default_suggestion | missed_routine_reminder",

      "current_segment_id": "<segment id where optimization/reminder is triggered>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "current_observation": "<what just happened that makes this a good optimization or reminder moment>",
      "current_local_context": "<short description of current scene>",
      "activation_type": "cross_day_pattern | expected_pattern_missing",
      "comparison_basis": "<how this day's pattern compares to earlier days / sessions>",
      "activation_reason": "<why optimization or reminder is considered helpful now>",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant proactive pattern-level optimization or gentle reminder suggestion>"
        },
        {
          "role": "user",
          "utterance": "<likely user reply (>= 12 words)>"
        }
        // optionally 1-2 more turns, alternating assistant ↔ user
      ]
    }
  ]
}

⸻

### ***What counts as a Routine Optimization signal?***

You care about recurring convenience patterns, not one-off tasks.

Positive signals (examples):
	1.	Multi-step routines (time_structured_routine / task_structured_routine)
		•	Similar sequence of steps in similar context:
		•	Afternoon cleaning: tidy room → collect trash → wipe surfaces → put tools back.
		•	Pre-meeting: open slides → check link → test mic → adjust seating.
		•	After cooking: clear table → rinse dishes → load dishwasher → wipe counters.
	2.	Configuration preferences (configuration_preference)
		•	Repeatedly setting the same or very similar configuration in a recurring context:
		•	Before sleep: AC around 26°C + bedside lamp dim + phone to silent.
		•	During meetings: phone on vibrate + laptop volume low.
	3.	Expected-but-missing routines (reminder candidates)
		•	From historical data, a routine_key appears on multiple days in a similar time-of-day window (e.g., 15:00-16:00 cleaning).
		•	In the current batch, time has progressed into or beyond this typical window, but:
		•	No occurrence of this routine_key has been observed so far today, and
		•	The user is engaged in other activities (with i_do_steps / speakers_say entries) that indicate the routine may have been forgotten or postponed.

In such cases, a soft reminder is a valid Routine Optimization trigger, because it helps maintain or intentionally reschedule a stable routine.

You should NOT treat as Routine Optimization:
	•	Pure safety behavior (→ Safety Proactive Service).
	•	Pure "getting better at a skill" without a configuration/routine optimization angle (→ Personal Progress Feedback).
	•	One-off or rare tasks with no sign of recurrence.

Simple test:
	•	If the main value is "this pattern happens again and again, and could be streamlined / auto-configured / gently remembered when missing" → candidate for Routine Optimization.
	•	If the main value is "I'm getting better at this skill" → likely belongs to Personal Progress.

⸻

### **Routine Key Granularity Rules**

Each routine_key should capture one pattern that you can describe with:
	•	For configuration_preference:
"When I am in [this situation/time], I tend to set/use [object] with [this configuration/mode/place]."
	•	For time_structured_routine / task_structured_routine:
"When I am in [this time/situation], I tend to do [step A → step B → step C] as one routine block."

Bad examples (too broad / vague):
	•	"General productivity routine"
	•	"Daily cleaning"
	•	"Charging everything"

Good examples:
	•	"Before sleep: set bedroom AC to ~26°C and dim bedside lamp to low warm light."
	•	"Before group meetings: open slides, check online link, and test microphone."
	•	"After cooking dinner: wipe counters, wash dishes, and take trash out."
	•	"During meetings: keep phone on vibrate and laptop volume low."

If you cannot write such a sentence without being vague, the key is too broad and should be split.

⸻

### **Historical Integration Rules**

1. When to reuse an existing routine_key
Given a new occurrence in the current batch and an existing routine_key:

Reuse that key (and its routine_id) only if:
	1.	Type alignment
		•	Same routine_type (configuration_preference vs time_structured_routine vs task_structured_routine).
	2.	Context alignment
		•	Similar time/situation trigger (e.g., "before sleep", "before meetings", "after cooking").
	3.	Configuration / task-cluster alignment
		•	For configuration_preference: similar object/setting and value/mode/place.
		•	For routines: similar set of tasks in similar order/structure.
	4.	Role alignment
		•	Plays the same role (prep / wrap-up / typical configuration).

If these align:
	•	Reuse the same routine_id and routine_key.
	•	Append a new occurrence with updated historical_context and routine_evolution.

2. When to create a NEW routine_key
Create a new routine_id / routine_key when:
	•	The trigger/context is fundamentally different.
	•	The core object/setting is different (for configuration preferences).
	•	The task cluster is different (for routines).
	•	Mixing them would make the key too broad, ambiguous, or internally inconsistent.

⸻

### **Objective**

For the current batch, with optional historical routine memory, you must:
	1.	Update pattern state (routine_state_updates)
		1.	Scan all segments and all interaction_records to find every time_window that reflects a possible:
			•	configuration-level preference, and/or
			•	time-structured routine, and/or
			•	task-structured routine,
		applying:
			•	Fine-Grained Evidence Priority, and
			•	Routine Key Granularity Rules.
		Prefer high recall; uncertain cases can be kept with lower occurrence_confidence.
		2.	Compare with routine_events in the Historical Routine Summary JSON (if provided):
			•	Each routine_key is one pattern (configuration or routine).
			•	The occurrences are concrete events.
			•	When scanning the current batch, if you detect a new event that matches an existing pattern (according to type + trigger/context + configuration/task cluster + role), you must:
			•	reuse that routine_key and its event_id in your output, and
			•	only add new occurrences from the current batch.
			•	If you detect a new pattern type not covered by any existing routine_key, you must:
				•	create a new candidate_events entry with:
				•	a new routine_key, and
				•	a newly assigned event_id that continues the numbering.
			•	For this new pattern, occurrences should contain only events from the current batch, and historical_context / routine_evolution should clearly state that this is a newly proposed pattern.
		3.	For each occurrence, record:
			•	observation, time_window, segment_id, supporting_source
			•	Contextual fields:
			•	local_context, historical_context, workflow_position, social_dynamics, routine_evolution, implicit_avoidance (optional), occurrence_confidence.
		4.	For each routine_key, summarize:
			•	how this batch extends or refines the pattern (routine_summary, routine_evolution), and
			•	overall batch_consistency_level and batch_confidence for this batch, taking historical memory into account when applicable.

	2.	Decide current-batch optimization triggers (optimization_triggers) and generate dialogues
		•	For each routine_key, use historical + current occurrences to determine whether this batch contains a moment where:
		•	There is enough longitudinal evidence (cross-day / cross-session) to justify optimization or reminder.
		•	The current occurrence is at a natural decision point:
			•	Pattern is being executed now, or
			•	Pattern is conspicuously missing/late in its usual time window, and the user is doing something else.
		•	For each such situation, activate one optimization trigger and generate a short multi-turn dialogue.

⸻

### **Activation Logic for Optimization Triggers**

A. Cross-day requirement (for stable pattern detection)
Before any activation:
	1.	For each routine_key, collect all its occurrences (historical + current batch).
	2.	Parse day_id from the DAY# prefix of each time_window.
	3.	Determine whether there are at least two distinct days where this routine_key appears.

A routine_key is eligible for proactive optimization only if:
	•	It appears on ≥ 2 different days (e.g., DAY1 and DAY2).

If it appears only on a single day:
	•	You may still record it in routine_state_updates,
	•	But DO NOT create any optimization_triggers for it in this batch.

⸻

B. Per-session uniqueness when pattern is present ("at most once per 2 hours per routine_key")
For present pattern optimization triggers (i.e.,
	•	multi-step-routines,
	•	configuration-only patterns),

we allow multiple activations on the same day, but **enforce a 2-hour cooldown for each routine_key.**

For each eligible routine_key, and for each day where it appears in the current batch:
	1.	Collect all occurrences of this routine_key on that day.
	2.	Sort them by the start time of their time_window ascending.
	3.	Partition them into session blocks on that day, where:
		•	Two consecutive occurrences belong to the same session block if the time gap between their start times is < 2 hours.
		•	If the gap is ≥ 2 hours, start a new session block.
	4.	For each session block, choose exactly one occurrence as the candidate optimization point, using this rule:
		1.	If any occurrence has workflow_position = "start", choose the earliest such occurrence in this block.
		2.	Else, if any occurrence has workflow_position = "end", choose the earliest such occurrence in this block.
		3.	Else, choose the earliest occurrence in this block.
	5.	That occurrence is the per-session candidate for this routine_key; other occurrences in the same 2-hour block are not activated.

This means:
	•	For multi-step-routines and configuration-only patterns,
	•	The same routine_key can be activated at most once every 2 hours on a given day.

⸻

C. Expected-but-missing routines (reminder candidates, strictly once per day)
For expected-but-missing routine reminders (missed_routine_reminder), we keep a once-per-day rule.

For each eligible routine_key, you must also check whether:
	1.	There are occurrences on ≥ 2 earlier days in similar time-of-day windows
(e.g., multiple days with occurrences between 15:00-16:00).
	2.	On the current day (DAY#) covered by this batch:
		•	Time (based on i_do_steps / speakers_say time_windows) has advanced into or clearly beyond the typical window, and
		•	No occurrence of this routine_key has been detected so far on this day.

	In this case, you may create a missed routine reminder trigger, with:
		•	trigger_type = "missed_routine_reminder"
		•	activation_type = "expected_pattern_missing"

	You must still choose a current_time_window from:
		•	The earliest i_do_steps[].time_window or speakers_say[].time_window that:
		•	Occurs after the typical start of the usual routine window on this day, and
		•	Represents a reasonable moment for a gentle check-in (the user is engaged in something else and the routine has likely been forgotten or delayed).

	Additional constraint for C-type triggers:
		•	For each routine_key, you may activate at most one missed_routine_reminder per day,
	regardless of how long the batch covers that day.

If such a moment cannot be clearly identified (e.g., the batch does not extend far enough into the usual time window), you should not trigger a reminder.

⸻

D. Per-occurrence activation criteria
For any candidate optimization/reminder point (present pattern or expected-but-missing):

Mark it as activated (i.e., create an optimization_triggers item) only if:
	1.	There exists at least one earlier day of the same routine_key to compare with.
	2.	It fits one of the following:
		•	Present pattern optimization (multi-step-routines / configuration-only):
			•	The current occurrence is the unique per-session candidate for its 2-hour block under rule B, and
			•	**No other optimization trigger for the same routine_key has been activated in the preceding 2 hours of clock time on that day, and**
			•	local_context, observation, and routine_type show a clear routine or configuration being executed now, and
			•	A meaningful optimization (bundling steps, saving defaults, pre-setup, reordering) can help future runs.
		•	Expected-but-missing reminder (missed_routine_reminder):
			•	Conditions in rule C hold
			•	stable pattern in similar time windows on earlier days, missing so far today, and a reasonable anchor activity exists), and
			•	No other missed_routine_reminder for the same routine_key has been activated yet on this day, and
			•	A gentle reminder or rescheduling suggestion would be helpful and not intrusive.
	3.	The situation is not primarily about safety or another proactive service type.

***The "historical_optimization_triggers" field in the Historical Routine Summary JSON is used to provide decision-making evidence for determining the Optimization Triggers in the current batch. (the gap between triggers >= 2 hours each type)***

⸻

#### Timestamp Rules for Optimization Triggers (strict)

For every item in optimization_triggers:
	•	current_time_window MUST be copied exactly from:
	•	i_do_steps[].time_window or
	•	speakers_say[].time_window.

Even for missed routine reminders, you must anchor to a real i_do_steps or speakers_say entry in the current batch (the activity during which the assistant decides to check in).

The assistant's first proactive utterance is interpreted as happening immediately after this current_time_window.

⸻

### **Dialogue Generation Objective (per activated trigger)**

For each activated occurrence in optimization_triggers:
	1.	Grounding
	Use:
		•	routine_key, routine_summary, and this trigger's:
		•	current_observation, current_local_context,
		•	historical_context, inferred_motivation,
		•	workflow_position, routine_evolution,
		•	comparison_basis, activation_type, trigger_type,
		•	plus earlier days' occurrences as evidence.

	2.	Optimization / Reminder content

	Regardless of trigger type, the proactive message must:
		1.	Recognize the pattern
			•	For present routines: acknowledge the recurring routine or configuration.
			•	For missed routines: gently note that the user usually handles X around this time without sounding like strict monitoring.
		2.	Provide convenience framing & emotional support
			•	Emphasize reduced friction, mental load, or remembering burden.
			•	For missed routines, validate that it's okay to change plans, and position the assistant as helpful, not scolding.
		3.	Offer concrete, realistic options
			•	For present routines:
			•	Bundle steps into a reminder/shortcut/checklist,
			•	Save / reuse as default configuration,
			•	Suggest pre-setup for tools / environment next time.
			•	For missed routines:
			•	Ask if the user wants to:
			•	still do it now with a quick checklist,
			•	move it to later today,
			•	skip it this time, or
			•	adjust the future reminder rule (e.g., weekdays only).

	Avoid replies that only paraphrase what the user did or usually does; always add:
		•	Pattern-level framing, and
		•	Future-facing, actionable optimization or reminder options.

	3.	Tone
		•	Practical, supportive, and non-judgmental.
		•	Especially for missed routines: avoid guilt; offer choices.
		•	Do NOT mention videos, annotations, or models.
		•	Do NOT talk explicitly about "cross-day analysis" or internal logs.

	4.	Dialogue structure

	For each optimization_triggers item:
		•	At least 2-3 turns:
		1.	Assistant (Turn 1)
			•	Proactive optimization / reminder suggestion, grounded in cross-day pattern.
		2.	User (Turn 2)
			•	Natural reply (≥ 12 English words).
			•	May accept, partially accept, postpone, decline, or adjust rules.
		3.	Assistant (optional Turn 3)
			•	Acknowledge the user's choice and either:
			•	confirm the new shortcut / checklist / default, or
			•	confirm the rescheduling / skipping rule, or
			•	back off politely.
		4.	User (optional Turn 4)
			•	Short acknowledgement or final confirmation.

	5.	Diversity
	Across all dialogues in one output:
		•	Vary phrasing:
		•	"you usually follow this routine…"
		•	"you often handle this situation in a similar way…"
		•	"you tend to set things up like this when…"
		•	Vary structure:
		•	Sometimes start from the pattern ("Since you often…")
		•	Sometimes from the offer ("If you'd like, I can…").
		•	Not every suggestion must be accepted; some users may say "not today", "only on weekdays", etc. Adapt accordingly.

⸻

### **Output Format (single JSON with two sections)**

Your output MUST be ONE JSON object:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Routine Optimization Proactive Service",

  "routine_state_updates": [
    {
      "routine_id": "rtopt_001",
      "routine_key": "<one routine/configuration pattern>",
      "routine_type": "configuration_preference | time_structured_routine | task_structured_routine",
      "routine_summary": "<how this pattern looks so far including this batch>",
      "batch_consistency_level": "high | medium | low",
      "batch_confidence": 0.0,

      "occurrences": [
        {
          "segment_id": "<source segment id>",
          "day_id": 2,
          "time_window": "DAY2 HH:MM:SS-HH:MM:SS",
          "supporting_source": "i_do_steps | interactions_with_objects | interactions_with_people | speakers_say | interaction_record",

          "observation": "<specific evidence of this pattern in this batch>",
          "local_context": "<what the user is doing / environment>",
          "historical_context": "<how this relates to earlier occurrences under this routine_id>",
          "inferred_motivation": "<behavior-grounded guess>",
          "workflow_position": "start | middle | end | standalone",
          "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
          "implicit_avoidance": "<optional>",
          "routine_evolution": "<optional>",
          "occurrence_confidence": 0.0
        }
      ]
    }
  ],

  "optimization_triggers": [
    {
      "trigger_id": "ropt_001",
      "routine_id": "rtopt_001",
      "routine_key": "<copied from routine_state_updates>",
      "routine_type": "configuration_preference | time_structured_routine | task_structured_routine",
      "trigger_type": "routine_block_optimization | configuration_default_suggestion | missed_routine_reminder",

      "current_segment_id": "<segment id where optimization/reminder is triggered>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "current_observation": "<what just happened that makes this a good optimization or reminder moment>",
      "current_local_context": "<short description of current scene>",
      "activation_type": "cross_day_pattern | expected_pattern_missing",
      "comparison_basis": "<how this day's pattern compares to earlier days / sessions>",
      "activation_reason": "<why optimization or reminder is considered helpful now>",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant proactive pattern-level optimization or gentle reminder suggestion>"
        },
        {
          "role": "user",
          "utterance": "<likely user reply (>= 12 words)>"
        }
        // optionally 1-2 more turns, alternating assistant ↔ user
      ]
    }
  ]
}

Notes:
	•	routine_state_updates only needs to include new or updated patterns from this batch (not the whole historical store).
	•	For the very first batch with no history:
		•	routine_summary: "initial hypothesis of a routine/configuration pattern in this batch."
		•	historical_context in each occurrence: "no prior records; first detected in this batch."
	•	***Do not repeatedly activate the same event or generate identical dialogue content.***

#### Failure Case Format
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Routine Optimization Proactive Service",
  "routine_state_updates": [],
  "optimization_triggers": [],
  "note": "No stable routine/configuration patterns or optimization-worthy cross-day moments were detected in this batch, given the current annotations and historical summary."
}
"""