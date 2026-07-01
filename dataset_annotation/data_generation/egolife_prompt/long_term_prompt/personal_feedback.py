PERSONAL_FEEDBACK_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (less than 2.5 hours).

Your job in this stage is signal mining and proactive feedback for a special Long-Term Proactive Service subtype:

•	service_main_type: "Long-Term Proactive Service"
•	service_sub_type: "Personal Progress Feedback Proactive Service"
(short: Progress Feedback Service)

This service focuses on skill learning / personal goals / performance tracking across sessions.
Beyond tracking, the core goal of this service is to give the user meaningful, emotionally supportive feedback and concrete, small-scale suggestions.
Your feedback should
	•	briefly evaluate the user's progress or persistence,
	•	provide emotional value (encouragement, acknowledgement, or gentle normalization of difficulty), and
	•	propose 1-2 realistic, actionable suggestions for future attempts.
Merely restating what the user just did without evaluation or suggestions is not allowed.

Typical examples:
	•	The user's cooking steps become smoother or faster over multiple days.
	•	Study or homework tasks are completed more consistently.
	•	Repeated practice motions (e.g., dance, sports, instrument, lab technique) become more standard, fluent, or complete.

You must both:
	1.	Accumulate progress-related events into a structured state (for long-term tracking), and
	2.	Within the current batch, detect moments that deserve immediate progress feedback and generate a short multi-turn dialogue.

⸻

### **Core Signal Types**

Within this batch, you focus on two complementary types of events:
	1.	Type A - Progress evidence events (state accumulation)
    Concrete episodes that show practice, execution, completion, or qualitative change for a specific skill / goal / performance pattern.
    Examples:
        •	Practicing the same chopping / cutting technique during cooking.
        •	Repeating the same dance routine or exercise drill.
        •	Doing a similar homework / study task again.
        •	Running through the same lab / work procedure with more fluency.
	2.	Type B - Feedback-worthy moments (activation + dialogue)
    Specific episodes where, given previous occurrences of the same pattern, it is valuable for the assistant to:
        •	Summarize improvement,
        •	Highlight persistence or effort, or
        •	Offer constructive micro-feedback or next-step suggestions.

These Type B moments must be grounded in longitudinal comparison:
	•	Cross-day (DAY1 → DAY2/3/…), or
	•	Same-day but clearly separate sessions (typically ≥ 2 hours apart, or distinct blocks like "morning practice" vs "evening practice").

You must NOT guess new events, invent timestamps, or hallucinate behaviors not supported by the annotations.

⸻

### **Authoritative Timestamps**

Every fine-grained annotation entry has a time_window aligned to the original video, for example:
	•	i_do_steps[].time_window
	•	interactions_with_objects[].time_window
	•	interactions_with_people[].time_window
	•	speakers_say[].time_window

All time_window strings already have the format:

DAY# HH:MM:SS-HH:MM:SS

You MUST only copy these existing time_window values.
Do NOT interpolate, merge, extend, shrink, or invent new time ranges.

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

2. Historical progress summary JSON (optional memory)

You may also receive ONE historical JSON summarizing previously mined progress-related events and triggered feedback dialogues, with the following structure:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Personal Progress Feedback Proactive Service",
  "progress_events": [
    {
      "progress_id": "pprog_001",
      "progress_key": "<abstract description of one skill/goal/performance pattern>",
      "progress_type": "skill_learning | personal_goal | performance | habitual_task",
      "progress_summary": "<how this pattern has evolved over previous hours/days>",

      "occurrences": [
        {
          "segment_id": "<old segment id>",
          "day_id": 1,
          "time_window": "DAY1 HH:MM:SS-HH:MM:SS",
          "supporting_source": "i_do_steps | interactions_with_objects | interactions_with_people | speakers_say | interaction_record",

          "observation": "<what happened>",
          "local_context": "<local scene / surroundings>",
          "metrics": {
            "practice_minutes": 8.0,
            "attempt_count": 3,
            "quality_trend": "struggling | partial_success | clear_success | unclear"
          },
          "historical_context": "<how it related to earlier ones>",
          "effort_type": "practice | execution | review | planning",
          "outcome_quality": "improved | similar | worse | unclear",
          "workflow_position": "start | middle | end | standalone",
          "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
          "occurrence_confidence": 0.0
        }
      ]
    }
  ],
  
  "historical_feedback_triggers": [
    {
      "trigger_id": "pfeed_001",
      "progress_id": "pprog_001",
      "progress_key": "<copied from progress_state_updates>",
      "trigger_type": "cross_session_progress | sustained_effort | milestone_completion",

      "current_segment_id": "<segment id where feedback is triggered>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "current_observation": "<what just happened that makes this a good feedback moment>",
      "current_local_context": "<short description of current scene>",
      "comparison_basis": "<how this compares to earlier runs / sessions>",
      "trigger_reason": "<why feedback is considered helpful now>",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant proactive feedback, referencing cross-session progress>"
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
	•	progress_key must describe one concrete skill/goal/performance area (e.g., "chopping vegetables for cooking", "evening English homework session", "repeating the same dance routine"), not a vague category like "study" or "exercise".
	•	progress_id is a stable identifier for that progress_key across days.
	•	day_id is derived from the DAY# prefix of time_window.

If the historical JSON is missing or progress_events is empty, treat this as no prior progress memory.

⸻

### **Fine-Grained Evidence Priority (critical)**

When detecting candidate occurrences, you must prioritize fine-grained entries before falling back to coarse summaries:

Primary sources for progress-related events
	•	i_do_steps
	•	interactions_with_objects
	•	interactions_with_people
	•	speakers_say

→ Treat each fine-grained entry (or short, coherent cluster within the same context) as a potential practice/execution episode if it clearly involves:
	•	repeating or performing a recognisable skill or task,
	•	making progress toward a personal goal, or
	•	showing a change in fluency / quality / completion status.

Timestamp rules for occurrences
	•	For every occurrence in progress_state_updates, you must copy one existing time_window directly from
        i_do_steps, interactions_with_objects, interactions_with_people, or speakers_say.
	•	You must never use interaction_record.time_window for occurrences.
	•	You must never merge, split, or invent time_window ranges.

Additional strict rule for feedback dialogue triggers (see below):
	•	For feedback_triggers.current_time_window, you may only copy from:
	•	i_do_steps[].time_window or
	•	speakers_say[].time_window.

⸻

### **What counts as Personal Progress Feedback signal?**

You care about learning curves and progress trajectories, not just single successes.

Positive signals (examples):
	•	Repeated attempts at the same skill with signs of:
	•	smoother execution,
	•	fewer hesitations / corrections,
	•	better timing or coordination,
	•	higher completion rate.
	•	Consistent continuation of a personal goal:
	•	regular study/homework sessions,
	•	repeated practice of a dance move,
	•	ongoing work on the same project steps.
	•	Explicit self-reflection or evaluation in speech:
        •	"I'm getting faster at this now."
        •	"This time the movement felt more stable."
        •	"Today I finally finished the whole sequence."

You should NOT treat as Progress Feedback:
	•	Pure safety behaviors (→ Safety Proactive Service).
	•	Pure configuration preferences without a learning/progress angle (→ Routine Optimization / Preference).
	•	Generic health habits like exercising or stretching without clear skill/goal tracking (→ Habit-Coaching).
	•	One-off tasks with no sign of repetition or progress trajectory.

Simple test:
	•	If the main value is "I'm getting better / more consistent at this over time" → candidate for Progress Feedback.
	•	If the main value is "this is just done or configured now" → likely belongs to another service.

⸻

### **Progress Key Granularity Rules**

Each progress_key should capture one skill/goal/performance pattern that you can describe with:

"When I practice or perform [X] in [this context] across sessions, my fluency/completion/quality can improve and deserves feedback."

Bad examples (too broad / vague):
	•	"General Learning"
	•	"Work Tasks"
	•	"Daily Exercise"

Good examples:
	•	"Chopping vegetables for cooking dinner in the kitchen"
	•	"Evening English homework sessions at the desk"
	•	"Practicing the same hip-hop dance routine in the living room"
	•	"Running through the same instrument warm-up sequence before practice"

If you cannot write such a single sentence without becoming vague, the key is too broad and should be split.

⸻

### **Historical Integration Rules**

1. When to reuse an existing progress_key
Given a new occurrence in the current batch and an existing progress_key from history, reuse that key (and its progress_id) only if:
	1.	Skill/goal alignment
	    •	It is clearly the same skill / goal / task (e.g., same dance routine, same type of homework, same cooking procedure).
	2.	Context alignment
	    •	Similar physical or situational context (same room / location type / device, same "before dinner" study block, etc.).
	3.	Role alignment
        •	The occurrence plays a compatible role:
        •	practice vs full execution vs review,
        •	similar place in the workflow (full run, partial drill, cool-down review, etc.).

If these align:
	•	Reuse the same progress_id and progress_key.
	•	Add a new occurrence under it with updated historical_context, metrics, and outcome_quality.

2. When to create a NEW progress_key
Create a new progress_id / progress_key when:
	•	The skill/goal is fundamentally different.
	•	The context is clearly different (e.g., completely different activity).
	•	Mixing them would make the key too broad or confusing.

⸻

### **Objective**

For the current batch of segments, with optional historical progress memory, you must:

1. Update progress state (**progress_state_updates**)

   - Scan all segments and interaction_records to find every time window that reflects:
     - practice or execution for a specific skill/goal, and/or
     - visible changes in fluency, completeness, or quality.
   - **When defining or updating any progress_key / progress_id, you MUST strictly follow the rules in:**
     - **"Progress Key Granularity Rules"** (to decide how fine-grained each progress_key should be), and
     - **"Historical Integration Rules"** (to decide when to reuse an existing progress_id vs. when to create a new one).
   - Integrate with `progress_events` from history by:
     - reusing `progress_key` and `progress_id` when allowed by those rules, or
     - creating a new `progress_id` / `progress_key` when the rules say a new thread is needed.
   - For each `progress_id`, append new occurrences capturing:
     - `segment_id`, `day_id`, `time_window`, `supporting_source`,
     - `observation`, `local_context`,
     - `metrics` (approximate is fine),
     - `historical_context`,
     - `effort_type`, `outcome_quality`,
     - `workflow_position`, `social_dynamics`,
     - `occurrence_confidence`.

2. Decide current-batch feedback triggers (**feedback_triggers**) and generate dialogues

   - For each `progress_key`, use its **historical + current** occurrences to determine whether this batch contains a moment where:
     - there is enough longitudinal evidence (cross-day or cross-session) to justify feedback, and
     - the current occurrence sits at a natural reflection point (end of a run/session, or clear improvement/struggle).
   - **When deciding whether to activate a feedback trigger, and how to phrase the feedback, you MUST strictly follow:**
     - **"Activation Logic for Feedback Triggers"** (to decide *whether* to trigger and at *which* occurrence), and
     - **"Dialogue Generation Objective (per activated trigger)"** (to structure and style the multi-turn dialogue).
   - For each qualifying occurrence, activate **one** feedback trigger and generate a short multi-turn dialogue, grounded in the linked progress history.

⸻

### **Activation Logic for Feedback Triggers**

A. Cross-session requirement (longitudinal comparison)
Before any activation:
	1.	For each progress_key, collect all its occurrences (historical + current batch).
	2.	Parse day_id from the DAY# prefix of each time_window.
	3.	Determine whether there are at least two distinct sessions, defined as:
      •	Occurrences on different days, or
      •	Occurrences on the same day but clearly separated (timestamps differ by ≥ 2 hours or obviously "morning vs evening" etc.).

A progress_key is eligible for proactive feedback only if:
	•	It has occurrences in ≥ 2 sessions as defined above.

If it appears only within one short session:
	•	You may still record it in progress_state_updates,
	•	But do NOT create any feedback_triggers for it in this batch.

B. Per-session uniqueness ("once per progress_key per session")
For each eligible progress_key:
	•	For every distinct session (DAY# or separated block) in the current batch, do:
        1.	Collect all occurrences of this progress_key in that session.
        2.	Sort them by the start time of their time_window ascending.
        3.	Choose exactly one occurrence as the candidate feedback point using this rule:
          •	Prefer an occurrence with workflow_position = "end" (end of a practice run / session).
          •	If none, prefer the latest occurrence in that session.
        4.	That occurrence is the per-session candidate; others in that session are not activated.

C. Per-occurrence activation criteria
Mark a candidate occurrence as activated (i.e., create a feedback_triggers item) only if:
	1.	There exists at least one earlier session of the same progress_key to compare with (baseline / prior attempts).
	2.	The current occurrence is the per-session unique candidate under rule B.
	3.	local_context, observation, metrics, or outcome_quality give some evidence of:
        •	improvement (faster, smoother, more complete, more standard), or
        •	meaningful persistence (continuing efforts on the same challenging task), or
        •	a clear completion milestone (e.g., finishing a sequence that was previously partial).
	4.	A helpful feedback message can:
        •	summarize progress so far, and/or
        •	give constructive micro-feedback or next-step suggestions
    that would make sense to the user.

If the episode is:
	•	purely a failure with no visible learning, or
	•	too ambiguous to judge, or
	•	mainly about safety / configuration / generic routine rather than progress,

then activated = false (no feedback dialogue), even though the occurrence is still recorded for progress tracking.

D. Intra-day cooldown for the same progress_key
For each progress_key on the same DAY#:
	•	If a feedback_trigger has already been activated at time T for this progress_key,
	•	** then you MUST NOT activate another feedback_trigger for the same progress_key within the next 2 hours on that day. **
	•	Any candidate occurrence whose time_window starts < 2 hours after the last activated trigger on that DAY# must be skipped (activated = false), even if it satisfies A-C.

***The "historical_feedback_triggers" field in the Historical Progress Summary JSON is used as evidence for determining whether a progress event of the same type should be activated for the current batch. (the gap between triggers >= 2 hours each type)***

⸻

### **Timestamp Rules for Feedback Triggers (strict)**

For every item in feedback_triggers:
	•	current_time_window MUST be copied exactly from:
    •	i_do_steps[].time_window or
    •	speakers_say[].time_window.
You MUST NOT use interactions_with_objects or interactions_with_people as the source of current_time_window, even if they provide context.
If multiple i_do_steps / speakers_say entries are involved in the same progress episode, choose the earliest time_window where the current practice/execution begins (e.g., starting the run, starting the explained reflection).

The assistant's first proactive utterance is interpreted as happening immediately after this current_time_window.

⸻

### **Dialogue Generation Objective (per activated trigger)**

For each activated occurrence in feedback_triggers:
	1.	Grounding
        •	Use progress_key, progress_summary, and this occurrence's:
        •	current_observation, current_local_context,
        •	historical_context, metrics, outcome_quality,
        •	plus earlier sessions for comparison.

  2.	Feedback content
    You MUST go beyond describing what the user just did. Each proactive message must contain all three layers:
        •	Progress evaluation - briefly state what has changed or remained stable across sessions
        e.g., smoother / more fluent / more complete / more consistent, or "still struggling but making steady attempts".
        •	Emotional support - acknowledge the user's effort, difficulty, or persistence in a supportive way
        e.g., recognizing that they kept practicing, normalizing that improvement takes time.
        •	Concrete next-step suggestion - give 1-2 small, realistic suggestions for future attempts
        e.g., focusing on o ne sub-step next time, slowing down a tricky part, scheduling a short practice slot.

    It is not allowed to only paraphrase the user's actions without (1) progress evaluation, (2) emotional value, and (3) at least one actionable suggestion.

	3.	Tone
        •	Supportive, encouraging, realistic.
        •	Do not sound like grading or harsh evaluation.
        •	Do NOT mention videos, annotations, or models.
        •   Avoid replies that only repeat what the user just did; always add your own evaluation and forward-looking guidance.
	
 4.	  Dialogue structure
	  •	At least 2-3 turns:
        1.	Assistant (Turn 1)
        •	Proactive feedback + brief suggestion, grounded in longitudinal comparison.
        2.	User (Turn 2)
        •	Natural reply (≥ 12 English words).
        •	May accept, adjust, or downplay the feedback.
        3.	Assistant (optional Turn 3)
        •	Acknowledge the user's stance and refine or tone down the suggestion.
        4.	User (optional Turn 4)
        •	Short acknowledgement or final decision.
  
  5.	Diversity
    •	Vary phrasing: sometimes emphasize "you've been practicing this several times now…", sometimes "compared to earlier sessions, this looked more stable…", etc.
    •	Not every user must enthusiastically accept the feedback; some may say "I'm tired today, maybe later", etc.

⸻

### **Output Format (single JSON with two sections)**

Your output MUST be ONE JSON object:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Personal Progress Feedback Proactive Service",

  "progress_state_updates": [
    {
      "progress_id": "pprog_001",
      "progress_key": "<one skill/goal/performance pattern>",
      "progress_type": "skill_learning | personal_goal | performance | habitual_task",
      "progress_summary": "<how this pattern looks so far including this batch>",
      "batch_consistency_level": "high | medium | low",
      "batch_confidence": 0.0,

      "occurrences": [
        {
          "segment_id": "<source segment id>",
          "day_id": 2,
          "time_window": "DAY2 HH:MM:SS-HH:MM:SS",
          "supporting_source": "i_do_steps | interactions_with_objects | interactions_with_people | speakers_say",

          "observation": "<specific evidence of practice/execution in this batch>",
          "local_context": "<what the user is doing / environment>",
          "metrics": {
            "practice_minutes": 10.0,
            "attempt_count": 2,
            "quality_trend": "struggling | partial_success | clear_success | unclear"
          },
          "historical_context": "<how this relates to earlier occurrences under this progress_id>",
          "effort_type": "practice | execution | review | planning",
          "outcome_quality": "improved | similar | worse | unclear",
          "workflow_position": "start | middle | end | standalone",
          "social_dynamics": "self-initiated | reacting_to_others | jointly_decided",
          "occurrence_confidence": 0.0
        }
      ]
    }
  ],

  "feedback_triggers": [
    {
      "trigger_id": "pfeed_001",
      "progress_id": "pprog_001",
      "progress_key": "<copied from progress_state_updates>",
      "trigger_type": "cross_session_progress | sustained_effort | milestone_completion",

      "current_segment_id": "<segment id where feedback is triggered>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "current_observation": "<what just happened that makes this a good feedback moment>",
      "current_local_context": "<short description of current scene>",
      "comparison_basis": "<how this compares to earlier runs / sessions>",
      "trigger_reason": "<why feedback is considered helpful now>",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant proactive feedback, referencing cross-session progress>"
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
	•	progress_state_updates only needs to include new or updated progress patterns from this batch (not all history).
	•	For the very first batch with no history:
    •	progress_summary: "initial hypothesis of a progress-related pattern in this batch."
    •	historical_context in each occurrence: "no prior records; first detected in this batch."
  •	***Do not repeatedly activate the same event or generate identical dialogue content.***

⸻

Failure Case Format

If you find no progress-related occurrences in this batch, or no occurrence meets the activation rules for feedback:
{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Personal Progress Feedback Proactive Service",
  "progress_state_updates": [],
  "feedback_triggers": [],
  "note": "No personal-progress patterns or feedback-worthy cross-session moments were detected in this batch, given the current annotations and historical summary."
}
"""