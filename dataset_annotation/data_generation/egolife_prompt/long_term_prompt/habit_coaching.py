HABIT_COACHING_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (up to ~2.5 hours).

Your job in this stage is habit-signal mining and dialog generation for:

•	service_main_type: "Long-Term Proactive Service"
•	service_sub_type: "Habit-Coaching Proactive Service"

In this stage, your task is to mine unhealthy behavior signals within the current batch and, for each event that satisfies the activation rules, generate a short Habit-Coaching proactive dialogue.

The assistant here behaves like a health / productivity tracker:
	•	For some behaviors (e.g., prolonged sitting, continuous screen use, not drinking/eating),
    it should trigger within the same day once unhealthy accumulation exceeds a threshold.
	•	For other behaviors (e.g., frequently working very late at night),
    it should trigger based on a cross-day lifestyle pattern.

Therefore, you must distinguish two types of unhealthy behavior:
	1.	Type A - Intra-day accumulation events
    Unhealthy exposure within the same day that should trigger a coaching reminder once a threshold is reached.
    Examples:
        •	Sitting or desk-bound work continuously for ≥ 40-60 minutes without meaningful standing/walking.
        •	Long continuous or cumulatively excessive screen usage (phone / laptop / tablet) without sufficient breaks.
        •	No drinking or no meals for a long stretch while engaged in work/activities.
	2.	Type B - Cross-day lifestyle patterns
    Habits that only become clearly problematic when observed across days.
    Examples:
        •	Repeatedly working or using screens very late at night (e.g., past midnight) over multiple days.
        • Consistently eating lunch around noon on prior days, but on the current day still skipping lunch well past the usual time (e.g., after 1 PM).
        •	Other long-term patterns (if specified) that require next-day or cross-day coaching, e.g.:
        "If the user worked very late last night, remind them the next evening to go to bed earlier."

You will receive two inputs each time:
	1.	Current batch annotations (required) - multiple segment-level human annotations for this ≤2.5hours window.
	2.	Historical habit summary JSON (optional) - previously mined habit events and triggered habit-coaching dialogues across earlier hours/days.

You must NOT guess new events, invent timestamps, or hallucinate behaviors not supported by the annotations.

⸻

### **Authoritative Timestamps**

Every annotation entry has a time_window aligned to the original video, for example:
	•	i_do_steps[].time_window
  •	speakers_say[].time_window
	•	interactions_with_objects[].time_window
	•	interactions_with_people[].time_window

All time_window strings use the format:

DAY# HH:MM:SS-HH:MM:SS

You MUST only copy these existing time_window values.
Do NOT interpolate or invent new time ranges.

⸻

### **Input Human Annotations**

1. Current batch: segment-level episodic annotations

You are given MULTIPLE JSON objects, each representing a <=10-minute egocentric segment:
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
The coverage of this batch is defined by the min/max timestamps in these JSONs.

⸻

2. Historical habit summary JSON (optional memory)

You may receive ONE historical JSON summarizing previously mined habit-related events and triggered habit-coaching dialogues:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Habit-Coaching Proactive Service",
  "habit_events": [
    {
      "habit_id": "habit_001",
      "habit_key": "<abstract description of one unhealthy habit pattern>",
      "habit_type": "intra_day_health | cross_day_lifestyle | mixed",
      "threshold_description": "<what threshold defines 'unhealthy'>",
      "habit_summary": "<how this habit has appeared over previous hours/days>",

      "occurrences": [
        {
          "segment_id": "<old segment id>",
          "day_id": 1,
          "time_window": "DAY1 HH:MM:SS-HH:MM:SS",
          "supporting_source": "i_do_steps | interactions_with_objects | interactions_with_people | speakers_say",

          "observation": "<what happened>",
          "local_context": "<local scene / surroundings>",
          "metrics": {
            "sitting_minutes": 45.0,
            "screen_minutes": 60.0
          },
          "historical_context": "<how it related to earlier ones>",
          "inferred_role": "sub_threshold | near_threshold | threshold_cross | followup_check",
          "workflow_position": "start | middle | end | standalone",
          "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
          "occurrence_confidence": 0.0
        }
      ]
    }
  ],
  "historical_habit_triggers": [
    {
      "trigger_id": "htrig_001",
      "habit_id": "habit_001",
      "habit_key": "<copied from habit_state_updates>",
      "trigger_type": "intra_day_threshold | cross_day_pattern",

      "current_segment_id": "<segment id where reminder is triggered>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say ",

      "current_observation": "<what just happened that makes this a good coaching moment>",
      "current_local_context": "<short description of current scene>",
      "trigger_reason": "<why threshold is considered exceeded or cross-day pattern confirmed>",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant coaching reminder, like a health band: e.g., you've been sitting for a long time, want to stand or stretch?>"
        },
        {
          "role": "user",
          "utterance": "<likely user reply (>= 10-12 words), may accept/decline/adjust>"
        },
        {
          "role": "assistant",
          "utterance": "<follow-up: confirm reminder, suggest simple action, or gently back off>"
        }
      ]
    }
  ]
}

Notes:
    •  habit_key must describe ONE concrete unhealthy pattern and its intended coaching meaning.
    •  **habit_id is a stable identifier bound to one habit_key.  
        - All occurrences of the same unhealthy habit pattern MUST reuse the same habit_id and habit_key.  
        - habit_id MUST follow a simple global increasing scheme: `"habit_001"`, `"habit_002"`, `"habit_003"`, …  
        - When you need to create a NEW habit_key (a new kind of unhealthy habit), you MUST look at all existing habit_id in the historical JSON, find the largest index, and assign the next number (e.g., history has `habit_001` and `habit_002` → next new habit_id MUST be `habit_003`).  
        - You MUST NOT rename or renumber any existing habit_id.**
    •	habit_type:
        •	intra_day_health: same-day accumulation events like prolonged sitting / screen / no drinking/ no meals.
        •	cross_day_lifestyle: cross-day patterns related to irregular daily routines, such as frequently working very late.
    •	mixed: when both views are relevant.
    •	day_id is derived from time_window's DAY#.

If no historical summary JSON is provided or habit_events is empty, treat this as no prior habit memory.

⸻

Service Category
	•	service_main_type: Long-Term Proactive Service
	•	service_sub_type: Habit-Coaching Proactive Service

⸻

### **What counts as a Habit-Coaching signal?**

You focus ONLY on health / productivity habits, not on preferences or safety emergencies.

We distinguish:

Type A - Intra-day accumulation
Behaviors where continuous or cumulative exposure in the same day becomes unhealthy and should trigger a coaching reminder, similar to a smartwatch alert.

    Typical patterns (examples, not exhaustive):
        1.	Prolonged sitting / lack of movement
            •	Multiple i_do_steps / speakers_say entries showing:
            •	sitting at desk / working on laptop / watching TV,
            •	with no standing/walking / posture change over a long stretch.
            •	Threshold examples:
                •	sitting_minutes ≥ 40min without a break.
        2.	Continuous screen usage
            •	speakers_say or i_do_steps indicating "scrolling phone", "working on laptop", "watching videos" with no interruption.
            •	Threshold examples:
                •	screen_minutes ≥ 45min without substantial off-screen activity.
                •	accumulated screen_time ≥ 5h one day
                •	Very frequent screen checking in a short time window (e.g. picking up phone every few minutes).
        3.	No hydration / no meals for a long stretch
            •	Long block of work / study / moving around with no drinking/eating events.
            •	Threshold examples:
                •	no_drink_minutes ≥ 120min while awake and active.
                •	no_meal_hours ≥ 5h since last meal, with continuous activities.

    ***These are always judged within the current day.***
    We do not sum "30min sitting today + 30min tomorrow" to trigger Habit-Coaching.

Type B - Cross-day lifestyle patterns
Behaviors where the risk comes from repeated late-night / irregular rest patterns across days.

    Typical patterns (examples, not exhaustive):
        1.	Repeated late-night work / screen use
            •	On DAY N: annotations show active computer/phone use or work well past midnight or very late night.
            •	On DAY N+1: user still follows similar schedule (e.g., again working late, or clearly tired).
            •	Habit-Coaching should:
                •	On the following evening, remind the user to rest earlier because of last night's late session.
                •	Or, if late-night behavior persists across multiple days, strengthen the coaching.
            •   On DAY N: user usually eats lunch around 12:00 (e.g., ordering food at noon).
            •   On DAY N+1: it is already past 13:00 and the user still has not eaten lunch.
            •   Habit-Coaching should:
                •   During DAY N+1 early afternoon (e.g., 13:00-13:30), remind the user to eat lunch on time based on their usual pattern.
        2.	Other cross-day health patterns (if specified)
            •	e.g., consistently waking up very early after very late nights.
            •   e.g., persistently irregular meal schedules across days (e.g., lunch or dinner consistently much later than usual).
            (You should only handle what is clearly supported by historical JSON; do not invent new categories.)

⸻

### **Fine-Grained Evidence Priority (critical)**

When detecting candidate habit occurrences, prioritize fine-grained entries:
	•	i_do_steps
  •	speakers_say
	•	interactions_with_objects
	•	interactions_with_people
	
→ Use these to infer:
	•	whether the user is sitting / using screen / moving / drinking / eating / resting,
	•	how long they keep doing so in this batch.

This priority applies to:
	•	which content you use to measure accumulation or late-night behavior, and
	•	which time_window you copy for each occurrence.

⸻

### **Authoritative Time Rules for Triggers**

For habit_triggers (where you will generate dialogues):
    •	current_time_window MUST be copied from: i_do_steps[].time_window or speakers_say[].time_window
    •	current_time_window must correspond to:
        • For Type A (intra-day):
                - the first interval in THIS DAY where the unhealthy threshold is crossed
                (e.g., the moment continuous sitting reaches 40+ minutes, or no-drink
                duration reaches ~2 hours), while the user is still in that pattern.
        • For Type B (cross-day lifestyle):
            - a representative interval in the CURRENT BATCH on the new day where the
                SAME unhealthy pattern clearly continues again, such as:
                ▫ the first "working on laptop / using screen" interval tonight after a
                    previous very late-night work session, or
                ▫ the first main-meal interval that again happens very late in the day
                    after a history of late dinners.
            - In other words, pick the earliest fine-grained i_do_steps / speakers_say
                time_window in this batch where the cross-day lifestyle habit is
                obviously active again on this DAY#, and use it as current_time_window.

You MUST NOT:
	•	invent new or merged time ranges,
	•	extend or shorten existing time_window strings.

⸻

### **Habit Key Granularity Rules**

Each habit_key should describe one concrete unhealthy habit + its coaching goal, e.g.:
	•	"Sits at desk for >40 minutes without standing or walking."
	•	"Uses phone or laptop screens continuously for long stretches without eye breaks."
	•	"Often works on laptop past midnight on consecutive days."

Template you should be able to fill:

"When I [do X in this situation] for [too long / too often],
the assistant should coach me to [take Y healthier action]."

Bad examples (too vague):
	•	"Health issues"
	•	"Bad time management"
	•	"Screen usage" (without threshold / context)

Good examples:
	•	"Prolonged desk sitting (>40min) without standing during study/work time"
	•	"Late-night laptop work after midnight on multiple days"

If you cannot express all occurrences under one habit_key with a single, concrete unhealthy pattern and threshold, you must split into multiple habit_keys.

⸻

### **Historical Integration Rules**
	1.	Reusing an existing habit_key
    Given a new occurrence and an existing habit_key in history, reuse it if:
        1.	Same target behavior
            •	Same kind of activity: sitting vs screen vs no drinking vs late-night work.
        2.	Same health framing and threshold scale
            •	e.g., both are "sitting continuously for ~40+ minutes", not one for 10min and another for 4h.
        3.	Same habit type
            •	Both are intra_day_health events about breaks,
            •	or both are cross_day_lifestyle events about late-night work.

    If all align:
        •	Reuse the same habit_id and habit_key.
        •	Add a new occurrence with appropriate metrics (sitting_minutes, screen_minutes, etc.).
        •	Update habit_summary to reflect how this batch extends or reinforces the habit.

	2.	Creating a new habit_key
    Create a new habit_id / habit_key when:
        •	The behavior is clearly different (e.g., "no drinking" vs "prolonged sitting").
        •	The time scale and coaching strategy differ (intra-day vs cross-day).
        •	The existing keys would become too vague if forced to include this pattern.
    **When you create a new habit_id, you MUST:  
        • Use the global `"habit_XXX"` format (e.g., `habit_001`, `habit_002`, …).  
        • Scan all existing habit_id in the historical JSON, find the maximum numeric index, and assign the next integer (zero-padded to 3 digits).  
        • Example: if existing IDs are `habit_001` and `habit_002`, the next new habit_id MUST be `habit_003`.**
⸻

### **Objective**

For the current batch of segments (≤2.5 hours), with optional historical habits, you must:
	  1.	Update habit_state (habit_state_updates)
        •   Scan all segments and interaction_records to find:
            •   new evidence for cross-day lifestyle patterns (Type B), such as when the user actually goes to sleep, when they have lunch/dinner, or when late-night work extends well past midnight.
            •   optionally, you may record intra-day accumulation events (Type A) only when they are needed as context for later cross-day reasoning; Type A is not required to be fully logged here.
        •   When deciding whether to reuse an existing habit_key or create a new one, you MUST follow the **Habit Key Granularity Rules** strictly, so that each habit_key consistently represents one coherent habit or lifestyle pattern.
        •   When integrating new occurrences into historical habit_state, you MUST follow the **Historical Integration Rules** strictly, including:
            •   reusing habit_id / habit_key only when the semantics and temporal pattern match the existing habit definition, and
            •   creating a new habit_id / habit_key when the behavior differs in goal, context, or time-of-day pattern.
        •	  Integrate with habit_events from history by reusing or creating habit_keys accordingly.
        •	  For each habit_id, append new occurrences capturing:
            •	segment_id, day_id, time_window, supporting_source,
            •	observation, local_context, metrics (e.g., sitting_minutes),
            •	inferred_role (sub_threshold / threshold_cross / followup_check),
            •	occurrence_confidence.

    2. Decide current-batch habit triggers (habit_triggers covering both types)
        •	For Type A (intra-day short-horizon habits, gap between triggers >= 2 hours each type):
            •	Within the current annotation batch, ***scan all segments and interaction_records***, track the cumulative exposure to each habit_key (e.g., continuous sitting, continuous screen use, time since last drink, etc.).
            •	When the accumulation for a given habit_key crosses the unhealthy threshold inside this batch, select the earliest suitable interval as a trigger point where:
                •	the user is still in that pattern (still sitting, still using screen, still not drinking, etc.), and
                •	a gentle coaching reminder would still be timely and helpful.
            •	Trigger frequency constraint:
                • You may trigger more than once per day for different habit_keys (e.g., one prolonged-sitting reminder + one hydration reminder).
                •	But for the same habit_key, you MUST trigger at most once within any ≤2.5h batch (no more than one strong reminder per 2-hour window for that pattern).
        •	For Type B (cross-day lifestyle habits):
            •	Use the "habit_events" in historical JSON to detect cross-day lifestyle patterns, such as:
                •	sleep_too_late (e.g., still working / using screen past 01:00),
                •	eat_too_late (e.g., main meal consistently after a very late time),
                •	or other explicitly defined cross_day_lifestyle habit_keys.
            •	For each such habit_key:
                •	First, confirm from history that the user has repeatedly shown this pattern on previous days (e.g., multiple late nights, multiple late dinners).
                • In today's relevant batch (e.g., the evening work batch for sleep_too_late, or the dinner-time batch for eat_too_late), treat the first segment that matches the risky context as a trigger opportunity and use its fine-grained time_window as the current_time_window.
            •	Daily trigger constraint:
                •	For each cross_day_lifestyle habit_key (e.g., sleep_too_late, eat_too_late), **you MUST trigger at most one coaching reminder per DAY#.**
                • Different habit_keys may each trigger once on the same day (e.g., the user may receive one “sleeping too late” reminder and one “eating too late” reminder on the same day, but each habit_key triggers at most once).
        •	For every trigger (Type A or Type B), prepare:
            •	a structured habit_trigger record, and
            •	a short multi-turn coaching dialogue grounded in that current_time_window and the corresponding habit_key.
        •	***The "historical_habit_triggers" field in the Historical Habit Summary JSON is used to help determine whether the activation frequency for the same habit type complies with the above trigger rules (especially for type A, the gap between triggers >= 2 hours each type).***

⸻

### **Dialogue Generation Objective (per activated trigger)**

For every activated habit trigger (Type A or Type B), you must generate a short, natural, multi-turn proactive coaching dialogue grounded strictly in the chosen current_time_window. The dialogue should behave like a gentle health/productivity assistant, similar to a smartwatch or wellbeing coach.

Assistant - Turn 1 (required)
The assistant should:
	•	Refer to the specific unhealthy habit pattern reflected by habit_key (e.g., prolonged sitting, long continuous screen use, late-night work, late meal).
	•	Mention why this moment is appropriate for a gentle reminder (threshold crossed, or cross-day pattern continuing).
	•	Offer one simple, actionable suggestion (stand/stretch, rest eyes, drink water, take a short break, eat on time, wind down earlier).
	•	Be supportive and non-judgmental.

User - Turn 2 (required, ≥ 12 English words)
The user replies naturally, often acknowledging, explaining context, or tentatively agreeing.
Example behaviors:
	•	"Yes, I didn't notice how long I've been sitting."
	•	"I'm almost done, but I can take a short break."

Assistant - Turn 3 (optional but recommended)
A brief follow-up that:
	•	Acknowledges the user's response,
	•	Offers a small encouragement or adjustment,
	•	Or gently confirms the coaching suggestion.

Tone requirements
	•	Calm, friendly, supportive.
	•	No scolding, no medical claims, no pressure.
	•	Avoid absolute statements ("you must…"). Prefer soft phrasing ("you could…", "might help to…").
	•	Do not mention sensors, annotations, thresholds, or dataset internals.

Content constraints
	•	All details must come from the habit_key, current_observation, and time_window.
	•	No hallucinated behaviors, tools, or locations.
	•	The dialogue must stay strictly tied to the unhealthy habit detected in this trigger.

⸻

### **Output Format (two sections)**

Your output must be ONE JSON object:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Habit-Coaching Proactive Service",

  "habit_state_updates": [
    {
      "habit_id": "habit_001",
      "habit_key": "<one unhealthy habit pattern>",
      "habit_type": "intra_day_health | cross_day_lifestyle | mixed",
      "threshold_description": "<natural-language threshold>",
      "habit_summary": "<how this habit looks so far including this batch>",
      "batch_consistency_level": "high | medium | low",
      "batch_confidence": 0.0,

      "occurrences": [
        {
          "segment_id": "<source segment id>",
          "day_id": 2,
          "time_window": "DAY2 HH:MM:SS-HH:MM:SS",
          "supporting_source": "i_do_steps | speakers_say",

          "observation": "<specific evidence of this habit in this batch>",
          "local_context": "<what the user is doing / environment>",
          "metrics": {
            "sitting_minutes": 45.0,
            "screen_minutes": 0.0,
            "no_drink_minutes": 120.0
          },
          "historical_context": "<how this occurrence relates to earlier ones under this habit_id>",
          "inferred_role": "sub_threshold | threshold_cross | followup_check",
          "workflow_position": "start | middle | end | standalone",
          "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
          "occurrence_confidence": 0.0
        }
      ]
    }
  ],

  "habit_triggers": [
    {
      "trigger_id": "htrig_001",
      "habit_id": "habit_001",
      "habit_key": "<copied from habit_state_updates>",
      "trigger_type": "intra_day_threshold | cross_day_pattern",

      "current_segment_id": "<segment id where reminder is triggered>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say ",

      "current_observation": "<what just happened that makes this a good coaching moment>",
      "current_local_context": "<short description of current scene>",
      "trigger_reason": "<why threshold is considered exceeded or cross-day pattern confirmed>",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant coaching reminder, like a health band: e.g., you've been sitting for a long time, want to stand or stretch?>"
        },
        {
          "role": "user",
          "utterance": "<likely user reply (>= 10-12 words), may accept/decline/adjust>"
        },
        {
          "role": "assistant",
          "utterance": "<follow-up: confirm reminder, suggest simple action, or gently back off>"
        }
      ]
    }
  ]
}

#### Notes & Constraints
	•	habit_state_updates:
    •   In practice, habit_state_updates is mainly used for Type B cross_day_lifestyle patterns (e.g., sleep time, late-night work end time, main meal times). You do not need to exhaustively log every Type A intra_day event.
	•	Only needs to include new or updated habits/occurrences from this batch (not all history).
	•	For the very first batch with no history:
    •	habit_summary: "initial hypothesis of an unhealthy habit pattern in this batch."
    •	historical_context: "no prior records; first detected in this batch."
    •	habit_triggers: Each entry corresponds to one concrete coaching moment in this batch.
    •	trigger_type:
      •   intra_day_threshold: same-day accumulation (e.g., prolonged sitting, no drinking, long uninterrupted screen use).
      •   cross_day_pattern: e.g., reminding the user in the evening, “You went to bed very late yesterday — try to rest earlier tonight.”
	•	You must NOT invent time_window; copy from existing fine-grained entries as defined above.
	•	***Do not repeatedly activate the same event or generate identical dialogue content.***

⸻

#### Failure Case Format
If you find no unhealthy habit accumulations and no cross-day lifestyle patterns that justify a coaching reminder:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Habit-Coaching Proactive Service",
  "habit_state_updates": [],
  "habit_triggers": [],
  "note": "No unhealthy accumulation or cross-day lifestyle habit requiring coaching was detected in this batch, given the current annotations and historical summary."
}
"""