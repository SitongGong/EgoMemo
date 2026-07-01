TOOL_USE_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric human annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous short-horizon batch (less than 2.5 hours).

Your job in this stage is signal mining and multi-turn dialogue generation for:

Instant Proactive Service
Subtype: Tool Use Proactive Service (“Tool Use”)

This service focuses on detecting suboptimal tool-use technique that:
	•	occurs during ongoing execution of a concrete task,
	•	does not create immediate physical danger (otherwise → Safety),
	•	can be corrected by small micro-action adjustments (grip / angle / force / motion), and
	•	benefits from a short, supportive assistant suggestion.

Tool Use is strictly about how to do a tool-related action — not about:
	•	whether the workflow step is correct (Error-Recovery),
	•	whether the state is dangerous (Safety),
	•	whether the user should move on to the next step (Next-Step Guidance), or
	•	whether the user forgot to close something (Resource Reminder).

⸻

### **Purpose of Tool Use**

A Tool Use Proactive Service event is triggered when:
	•	The user is currently performing a correct step involving a specific physical tool, and
	•	Their grip / angle / stability / force / contact position / micro-motion is suboptimal, and
	•	The issue does not cause immediate serious injury risk (if yes → Safety), and
	•	The user's goal and workflow direction are correct (if wrong → Error-Recovery), and
	•	Fixing it requires only minor micro-action adjustments, not rollback or undo of previous steps.

This service should provide gentle, real-time technique optimization, similar to how an experienced coworker might say:
	•	“Hold it a little closer to the base—that'll be more stable.”
	•	“Your brush angle is quite flat; tilting it a bit will give better coverage.”
	•	“Your screwdriver grip looks a bit unsteady; try keeping your wrist straighter.”

⸻

### **Authoritative Timestamps & Fine-Grained Evidence Priority**

Each fine-grained annotation entry has a time_window aligned to the original video, such as:
	•	i_do_steps[].time_window
	•	speakers_say[].time_window

All time_window strings already have the format:

DAY# HH:MM:SS-HH:MM:SS

When you output current_time_window, you MUST:
	•	copy it only from:
	•	i_do_steps[].time_window, or
	•	speakers_say[].time_window.

You must NOT use:
	•	interactions_with_objects[].time_window,
	•	interactions_with_people[].time_window,
	•	interaction_records[].time_window

as the source of current_time_window (although they can still provide contextual evidence that a tool is being used, and how).

You must NOT:
	•	fabricate new timestamps,
	•	merge multiple ranges,
	•	extend or shrink an existing range.

Fine-grained evidence priority

When detecting Tool Use technique issues, you must prioritize fine-grained entries:
	•	Primary sources for discovering tool-use technique episodes:
	•	i_do_steps
	•	interactions_with_objects
	•	interactions_with_people
	•	speakers_say (if describing tool manipulation or technique)

You must tie each Tool Use event to one concrete fine-grained entry that clearly shows suboptimal technique. Do not infer from vague or long summaries alone.

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
	•	service_main_type: "Instant Proactive Service"
	•	service_sub_type: "Tool Use Proactive Service"

This service is distinct from:
	•	Error-Recovery (wrong object / step / configuration → requires rollback),
	•	Instant Safety (acute physical danger, e.g., cut / burn / shock / collision),
	•	Next-Step Guidance (what to do next in a workflow),
	•	Short-Term Resource Reminder (forgot to close / turn off / secure),
	•	Habit-Coaching / Long-Term (posture, long-term strain, lifestyle).

⸻

### **What counts as a Tool Use signal?**

A Tool Use signal is valid when all of the following hold:

1. A specific physical tool is involved
Examples:
	•	knife, spatula, ladle, tongs, peeler,
	•	screwdriver, wrench, drill, scissors,
	•	brush, sponge, mop, squeegee,
	•	cleaning tools, small handheld devices, etc.

2. The action itself is correct
	•	The user is performing the right task and right step in the workflow.
	•	The chosen tool is appropriate for the task (not obviously wrong tool type).
	•	The current workflow state remains valid even if technique is not ideal.

3. The micro-technique is suboptimal
Look for concrete issues such as:
	•	grip is too far from the active end, reducing stability,
	•	tool angle is too shallow / too steep relative to the surface,
	•	wrist seems unstable or twisted,
	•	two-handed stabilization would clearly help but user uses one hand only,
	•	too little or too much force is applied,
	•	contact surface is uneven or only partially engaged,
	•	motion path is inefficient (e.g., large arm swings instead of small controlled motions).

4. The issue is fixable by micro-adjustments
	•	No rollback, undo, or step reset is required.
	•	Adjusting grip, angle, stance, or motion within the current step is enough to improve the outcome.

5. No acute physical hazard (otherwise → Safety)
If the same technique pattern also creates immediate bodily danger (cut, burn, shock, severe slip, collision within the next ≈0-10 seconds), then for this service it must be handled as Instant Safety, not Tool Use.

⸻

### **Distinction From Other Service Types**

1. Tool Use vs. Error-Recovery (MOST IMPORTANT)
Error-Recovery occurs when:
	•	The user has made a workflow-level mistake:
	•	wrong object / wrong tool,
	•	wrong slot / wrong orientation,
	•	wrong step order or invalid configuration.
	•	The system is now in a wrong or invalid state.
	•	Fixing it requires rollback (undo/redo), not just better technique.
	•	Even perfect technique cannot fix the state.

Therefore, Tool Use MUST NOT be used when:
	•	The user selected the wrong item.
	•	The user inserted something into the wrong slot/orientation.
	•	A step must be undone before proceeding.
	•	The state is invalid even if technique becomes perfect.

Diagnostic question:

“If the user's technique becomes perfect from this moment on, will the underlying problem disappear?”

	•	Yes → Tool Use is possible (as long as state is valid).
	•	No → This is Error-Recovery, not Tool Use.

Examples of Error-Recovery (NOT Tool Use):
	•	Upside-down battery inserted into a device (must remove & reinsert correctly).
	•	Wrong container used for a chemical step.
	•	Clearly wrong tool selected for the target (e.g., flat screwdriver in a Phillips screw).
	•	A component fully misaligned in a way that cannot be fixed by grip/angle alone.

⸻

2. Tool Use vs. Safety
Safety applies when:
	•	The technique issue causes immediate physical danger (cut, burn, slip, shock, collision) within the next ≈0-10 seconds.

If danger exists, then → ALWAYS Safety, NEVER Tool Use.

Examples that become Safety:
	•	Knife held close enough to cut the user's finger within a small motion.
	•	Hot pot / boiling liquid extremely close to bare skin.
	•	User touching or approaching exposed live wires.
	•	Heavy object visibly unstable and about to fall on or near the user.

⸻

3. Tool Use vs. Next-Step Guidance
If the user:
	•	is doing the correct step, but
	•	mainly needs help with what to do next in the sequence,

→ this is Next-Step Guidance, not Tool Use.

Tool Use = how to do the current tool action better.
Next-Step Guidance = what to do after this step.

⸻

4. Tool Use vs. Short-Term Resource Reminder
If the main issue is:
	•	forgetting to turn off a stove or power switch,
	•	leaving water running,
	•	leaving a tool powered on or a lid open,

→ this is Short-Term Resource Reminder, not Tool Use.

⸻

5. Tool Use vs. Habit-Coaching / Ergonomics
Long-term ergonomic concerns (e.g., posture at a desk, long-term strain, monitor distance) without a specific tool-use micro-technique in a concrete step → Habit-Coaching / Long-Term, not Tool Use.

Tool Use only addresses immediate technique optimization during an ongoing tool operation.

⸻

### **Short-Horizon Technique Window**

Tool Use operates strictly within an instant, ultra-short-horizon window, reflecting local technique within an ongoing step.

The model must follow these temporal constraints:
	•	Immediate technique window (≈0-10 seconds)
	•	A Tool Use opportunity arises when the user is currently manipulating a tool,
	•	and the suboptimal technique is visible now or within the next few seconds.
	•	The assistant's suggestion is assumed to be delivered during the same short action phase, not minutes later.
	•	No long-horizon technique memory
	•	Do not reason over tool-use patterns older than ~10 seconds as evidence for a Tool Use event.
	•	Do not chain across multiple distant clips or sessions.
	•	Tool Use must rely on local, immediate evidence within one short micro-episode.

⸻

### **Time-Window Rules**

For every Tool Use event, you must define a single:
	•	current_time_window representing the short interval where the suboptimal technique occurs and where the assistant speaks up.

Rules:
	1.	Source of current_time_window
        •	You MUST copy current_time_window from exactly one of:
        •	i_do_steps[].time_window, or
        •	speakers_say[].time_window.
        •	You must NOT use:
        •	interactions_with_objects[].time_window,
        •	interactions_with_people[].time_window,
        •	interaction_records[].time_window
    as the timestamp source for current_time_window.
	2.	Single moment only
        •	You must copy one existing time_window string exactly.
        •	You must NOT:
        •	invent new time ranges,
        •	merge two ranges,
        •	extend or shorten an existing range.

The assistant's first proactive utterance is interpreted as happening during or immediately after this current_time_window.

⸻

### **Tool Use Detection Objective**

For the given batch:
	1.	Scan all fine-grained entries (i_do_steps, interactions_with_objects, interactions_with_people, speakers_say) to identify episodes where the user is clearly manipulating a tool.
	2.	For each candidate episode, check whether:
        •	The task and step are correct (no wrong-object / wrong-slot / wrong-order error).
        •	There is no acute physical hazard (otherwise → Safety).
        •	The main issue is a technique inefficiency (grip, angle, stability, force, motion, contact).
        •	Technique issues can be fixed by local micro-adjustments, with no rollback.
	3.	Filter out episodes where a different service is dominant:
        •	Safety (immediate bodily danger),
        •	Error-Recovery (invalid workflow state requiring undo),
        •	Short-Term Resource Reminder (forgotten closure),
        •	Next-Step Guidance (sequencing),
        •	Habit-Coaching / Long-Term (posture, lifestyle).
	4.	For each valid Tool Use episode:
        •	Select the most fine-grained i_do_steps or speakers_say entry that best captures the problematic technique moment.
        •	Copy its time_window as current_time_window.
        •	Record what tool is used, what the user is trying to do, and what technique issue appears.

⸻

### **Dialogue Generation Objective**

For each Tool Use event, you must generate a short, natural multi-turn dialogue.

Assistant (Turn 1)
	•	Speaks first.
	•	Mentions or clearly implies the specific tool.
	•	Points out the technique issue in a neutral, descriptive way.
	•	Offers 1-2 concrete micro-adjustments (e.g., grip, angle, force, motion).

Examples:
	•	“If you hold the knife a bit closer to the handle base, it will feel more stable while you slice.”
	•	“Your wrist is quite bent while using the screwdriver; straightening it a little can give you better control.”
	•	“Right now the brush is almost flat; tilting it slightly will help the bristles reach the surface more evenly.”

Do not mention “video”, “annotations”, “dataset”, or internal processing.

⸻

User (Turn 2)
	•	Provides a natural reply (≥ 12 English words).
	•	May:
	•	acknowledge the suggestion and adjust,
	•	ask for clarification,
	•	say they prefer their current style,
	•	briefly explain what they're trying to achieve.

⸻

Assistant (optional Turn 3)
	•	Acknowledges the user's reaction.
	•	Either:
	•	refines the suggestion (more concrete), or
	•	supports the user's choice and gently closes the guidance.

⸻

User (optional Turn 4)
	•	Short acknowledgment or confirmation (e.g., “Got it, I'll try that grip now.”).

Tone:
	•	Helpful, calm, non-judgmental.
	•	Emphasize making the task easier or more stable, not blaming.
	•	Avoid scolding (“you are doing it wrong”); prefer supportive phrasing:
	•	“You might find it easier if…”
	•	“It may feel more stable when…”

⸻

### **Output Format**

Your output MUST be exactly one JSON object:
{
  "service_main_type": "Instant Proactive Service",
  "service_sub_type": "Tool Use Proactive Service",

  "tool_use_events": [
    {
      "event_id": "tooluse_001",

      "current_segment_id": "<segment id>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "scene_description": "<neutral description of what the user is doing with the tool>",
      "trigger_reason": "<why this moment is a Tool Use technique issue (not Safety / not Error-Recovery)>",

      "tool_name": "<knife / screwdriver / brush / sponge / etc.>",
      "technique_issue": "<specific suboptimal technique (e.g., unstable wrist, angle too flat)>",
      "why_suboptimal": "<why this technique reduces stability, efficiency, or quality>",
      "risk_level": "low",
      "requires_rollback": false,
      "is_safe": true,

      "occurrence_confidence": 0.0,

      "dialogue": [
        {
          "role": "assistant",
          "utterance": "<concise, supportive technique advice>"
        },
        {
          "role": "user",
          "utterance": "<>=12-word natural reply>"
        }
        // optional extra turns (assistant ↔ user)
      ]
    }
  ]
}

Notes:
	•	tool_use_events may contain 0, 1, or multiple entries depending on how many Tool Use opportunities are detected.
	•	supporting_source MUST be "i_do_steps" or "speakers_say", consistent with the allowed sources for current_time_window.

⸻

#### Failure Case Format

If no Tool Use opportunity exists in the current batch:
{
  "service_main_type": "Instant Proactive Service",
  "service_sub_type": "Tool Use Proactive Service",
  "tool_use_events": [],
  "note": "No Tool Use technique issues were detected in the current batch, given the current annotations."
}
"""