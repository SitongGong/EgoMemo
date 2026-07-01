TASK_REMINDER_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (the actual span is determined by the timestamps in the input, typically within ≤ 2.5 hours).

Your job in this stage is signal mining and dialogue generation for a special Episodic Proactive Service subtype:

Episodic Task Reminder Proactive Service  
(short: Episodic Task Reminder)

This service focuses on **pending steps / tasks inside the same episodic window** (same local activity or short continuous session), where:

- The user has **started or committed to a concrete task or step** earlier in the SAME episodic context.
- There is **no clear evidence of completion** for that task/step.
- The user's behavior now shows they are **transitioning away to another activity or state**, so the task is likely to be left unfinished **right now**.
- The assistant should remind the user **at the moment of this transition**, before the task is silently dropped.

Typical examples:

- The user is doing a multi-step procedure (step 1, step 2…) and skips step 3, then starts packing up or leaving the lab.
- The user says "I'll submit this result once I finish checking it," then later closes the document and opens a different app without submitting.
- The user starts a short form, fills in most fields, then switches to another unrelated task while some required fields are still empty.

***You must:***
1. Scan the current batch to detect **within-episode tasks** that have **pending steps**.
2. For each such task, watch later moments in the **same episodic window** where the user appears to **move on without completing it**.
3. For each strong case, choose one current moment as the **reminder trigger** and generate a short multi-turn dialogue where the assistant:
   - points out the unfinished step or task, and
   - offers concrete, helpful options (finish now vs. schedule/ignore).

There is **no historical memory JSON** for this service:

- You treat each batch independently.
- You do **not** maintain cross-day or long-horizon memory.
- You only output dialogues for **this batch**, each grounded in:
  - one **pending task/step episode** (past), and
  - one **transition-away episode** (current).

You must NOT guess new events, invent timestamps, or hallucinate behaviors not supported by the annotations.

⸻

### **Authoritative Timestamps & Fine-Grained Evidence Priority**

Each fine-grained annotation entry has a time_window aligned to the original video, for example:

- i_do_steps[].time_window
- interactions_with_objects[].time_window
- interactions_with_people[].time_window
- speakers_say[].time_window

All time_window strings already have the format:

> DAY# HH:MM:SS-HH:MM:SS

You MUST only copy these existing time_window values.  
Do NOT interpolate, merge, extend, shrink, or invent new time ranges.

**Fine-grained evidence priority**

When detecting episodic task reminders, you must prioritize fine-grained entries:

- Primary sources to consider as candidate episodes:
  - i_do_steps
  - interactions_with_objects
  - interactions_with_people
  - speakers_say

Each **pending-task episode** and each **transition-away episode** must correspond to at least one fine-grained entry whose content supports that interpretation.

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

These interaction_records and their subfields are your **only evidence**.

The coverage of this batch is defined by the minimum and maximum timestamps appearing in these JSONs.  
You can assume this coverage is at most about 2.5 hours and usually contains several **episodic windows** (short continuous activities).

⸻

### **Service Category**

- service_main_type: "Long-Term Proactive Service"
- service_sub_type: "Episodic Task Reminder Proactive Service"

Note: this service operates on **short-horizon, same-episode pending tasks**; it is different from Episodic Memory Recall, which focuses on recalling earlier helpful episodes across a longer short-horizon (≤ 2h) even when the current activity is different.

⸻

### **What counts as an Episodic Task Reminder signal?**

An Episodic Task Reminder opportunity is defined by **one task thread** inside the same episodic window, with **two roles**:

1. A **pending-task episode** (earlier in this episode):
   - The user **starts**, **mentions**, or is **in the middle of** a clearly defined task or step, such as:
     - following a numbered or ordered procedure ("step 1 / step 2 / …"),
     - stating an intention to finish something within this episode ("before I leave I'll do X"),
     - performing most of a recognizable task (washing dishes, exporting a result, submitting a form).
   - There is **no subsequent evidence** in this batch that the critical step or finalization is completed.

2. A **transition-away episode** (later in the same episodic window):
   - The user **switches away** from that task thread, e.g.:
     - leaving the current location,
     - starting a different, unrelated activity,
     - closing the relevant app/document,
     - putting away tools or changing context.
   - At this transition moment, the earlier task/step is **still pending**.

The core question:  
> "Is there something the user clearly intended or needed to do in this episode, which now looks left unfinished as they move on?"

We distinguish three common reminder types:

1. **Skipped step in a multi-step procedure (skipped_step)**  
   - Past: annotations show a structured process (explicit "step 1 / step 2 / …" or implicit ordered actions) where a **key step is never executed**, e.g.:
     - user does step 1 and step 3 but there is no evidence of step 2,
     - user prepares a sample but never logs the result, etc.
   - Current: user transitions into another task or leaves the station (new task_transition, new context, or clear "packing up").
   - Reminder: "There's still step X you haven't done in this procedure."

2. **Unfinished task in this episode (unfinished_task)**  
   - Past: the user starts or states a **task with a clear completion condition**:
     - "I'll submit this result after checking it,"
     - opening a submission page, filling most fields,
     - beginning to clean an area or pack a box.
   - Current: there is no evidence that the completion event happened (no "submit", no final cleaning step, etc.), and the user is now:
     - idle in a different app,
     - walking away,
     - or starting a new unrelated task.
   - Reminder: "You haven't actually finished / submitted / wrapped up X yet."

3. **Pending check or confirmation within this episode (pending_check)**  
   - Past: the user sets up a **short verification step** inside this episode:
     - "After this run I need to verify the result,"
     - enabling a setting but intending to test it,
     - preparing an experiment whose outcome should be checked before leaving.
   - Current: the user:
     - finishes the main run and starts to move on,
     - or changes context without doing the check.
   - Reminder: "You planned to check Y before moving on; do you want to confirm it now?"

**Time & context constraints (critical):**

- The pending-task episode and the transition-away episode must:
  - be within the **same episodic window** (same local activity context),
  - occur within a short horizon (typically **minutes**, always ≤ 2 hours),
  - belong to the **same task thread** (same object/result/form/area/procedure).
- Very short gaps (a few seconds) that represent micro-steps or natural pauses inside the same continuous action should **not** trigger a reminder; treat these as normal task execution, not as "moving on".

Simple test:

- If the main value is "**you're about to walk away / switch tasks while something here is still unfinished**" → candidate for Episodic Task Reminder.
- If the main value is "remember something from earlier in this session even though the current activity is different" → Episodic Memory Recall.
- If the main value is "build a habit / routine / long-term progress" → other services (Habit-Coaching, Routine Optimization, Personal Progress Feedback, Long-Horizon Memory-Link).

⸻

### **Time-Window Rules (strict)**

For every Episodic Task Reminder dialogue you output, you must define:

- A **past_time_window** representing the pending-task episode.
- A **current_time_window** representing the moment when the assistant speaks up (transition-away).

Rules:

1. **Past time window (past_time_window)**

   - MUST be copied exactly from one fine-grained entry's time_window:
     - i_do_steps[].time_window, or
     - interactions_with_objects[].time_window, or
     - interactions_with_people[].time_window, or
     - speakers_say[].time_window.
   - Do NOT use interaction_record.time_window.
   - Do NOT merge or fabricate new ranges.

2. **Current time window (current_time_window)**

   - MUST be copied exactly from one:
     - i_do_steps[].time_window, or
     - speakers_say[].time_window.
   - You MUST NOT use interactions_with_objects or interactions_with_people as the source for current_time_window, even if they describe related context.
   - Choose the **earliest** time_window where:
     - the user's behavior first clearly shows they are **leaving or switching away** from the task thread, and
     - a reminder about the pending step would still be timely and helpful.

3. **Temporal order & horizon**

   - past_time_window MUST be strictly earlier than current_time_window.
   - The gap between them should be:
     - more than a few seconds (so it's not the same micro-action), and
     - at most ~2 hours, usually within the same short session.

The assistant's first proactive utterance is interpreted as happening **immediately after** this current_time_window.

⸻

### **Task Reminder Detection Objective**

For the current batch (no external history):

1. Scan all segments and interaction_records to find **candidate task threads**:
   - Sequences of i_do_steps / interactions / speech that:
     - share the same object, document, device, or area, and/or
     - are described as parts of one procedure or goal.

2. For each task thread, identify **critical steps or completion conditions**, such as:
   - explicit numbered steps,
   - explicit "I'll do X after Y" statements,
   - implicit but clear "finalization" events (submit, save, clean up, switch off, confirm).

3. For each critical step or completion condition:
   - Check whether there is **evidence of completion** later in the episode.
   - If **no** completion is detected, treat this as a **pending-task episode**, with:
     - its segment_id, past_time_window, past_observation.

4. For each pending-task episode, search **later in the same episodic window** for a **transition-away episode** where:
   - the user starts a **different task** or leaves the relevant context,
   - AND this happens **before** any completion evidence appears.

5. For each strong pair (pending-task episode, transition-away episode):
   - Ensure both episodes are supported by specific fine-grained annotations.
   - Ensure they belong to the **same task thread** and the gap fits the short-horizon nature.
   - Select **one** transition-away episode as the trigger point.

6. For each such pair, generate:
   - One structured reminder entry (with both past and current timestamps).
   - One multi-turn dialogue where the assistant gently reminds the user of the pending task.

You may output multiple task_reminders in one batch, but:

- Never duplicate the same (past, current) pair.
- Do not trigger multiple times for trivially overlapping transitions referring to the same pending task.

⸻

### **Dialogue Generation Objective**

For each activated episodic task reminder pair, you must generate a short multi-turn dialogue that:

1. **Grounds in the specific pending task**

   - The assistant should implicitly or explicitly reference the unfinished step or task, in a natural way, e.g.:
     - "You haven't done step 2 of this procedure yet."
     - "You haven't submitted this result you were working on."
     - "You planned to double-check this setting before leaving."
   - Do **not** mention "video", "annotations", "models", or internal logging.

2. **Connects to the transition-away behavior**

   - Make it clear that the reminder comes **because the user is moving on**, e.g.:
     - "Since you're putting things away now…"
     - "Now that you're switching to something else…"
     - "Before you leave the lab area…"

3. **Offers concrete, practical choices**

   - For **skipped_step**:
     - Suggest doing the missing step now, or explicitly skipping it with awareness.
   - For **unfinished_task**:
     - Suggest finishing now, or scheduling a reminder / marking it as deferred.
   - For **pending_check**:
     - Suggest doing the quick check now, or confirming that it can be safely delayed.

4. **Tone and emotional value**

   - Gentle, non-judgmental, supportive.
   - Emphasize reducing cognitive load:
     - "so it doesn't slip your mind,"
     - "just in case you still want to finish it now."
   - Avoid scolding language ("you forgot again"); instead:
     - "If you'd like, we can…"
     - "Do you want to quickly handle it before moving on?"

5. **Dialogue structure**

  Each proactive_dialogue should contain at least 2 turns, preferably 3-4:

  1. **Assistant (Turn 1)**  
    - Proactively mentions the pending task/step,
    - links it to the fact that the user is leaving / switching,
    - and offers one or two clear options.

  2. **User (Turn 2)**  
    - Natural reply (≥ 12 English words).
    - May:
      - accept and finish now,
      - partially accept (e.g., "remind me later"),
      - decline ("it's intentional, I'm done with it"),
      - or modify the reminder rule.

  3. **Assistant (optional Turn 3)**  
    - Acknowledge the user's choice and either:
      - confirm taking action now (e.g., guiding through the step),
      - confirm a follow-up reminder rule,
      - or gracefully back off if the user declines.

  4. **User (optional Turn 4)**  
    - Short acknowledgement or final confirmation.

  **Diversity**

  Across different reminders in the same output:

  - Vary how you phrase the transition:
    - "since you're leaving this desk…"
    - "as you switch to another task…"
    - "now that you're done here for the moment…"
  - Vary the help framing:
    - "Want to quickly finish X before you go?"
    - "Should we tick off that last step now?"
    - "I can set a reminder for later if you prefer."

Not every user must comply; some may say "not now," "I intentionally left it," or "remind me this evening."

⸻

### **Output Format**

Your output MUST be ONE JSON object:

{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Episodic Task Reminder Proactive Service",

  "task_reminders": [
    {
      "reminder_id": "etask_001",
      "reminder_type": "skipped_step | unfinished_task | pending_check",

      "task_thread_summary": "<short abstract description of the task this reminder belongs to>",

      "current_segment_id": "<segment id where the assistant speaks>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "current_supporting_source": "i_do_steps | speakers_say",
      "current_observation": "<what is happening now that shows the user is moving away>",
      "current_local_context": "<short description of current scene / situation>",

      "past_segment_id": "<segment id of the pending-task episode>",
      "past_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "past_supporting_source": "i_do_steps | speakers_say",
      "past_observation": "<what happened earlier that defines the unfinished task or step>",

      "time_gap_seconds": 0.0,
      "transition_reason": "<why this looks like a switch away from the task>",
      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant proactive reminder, linking pending step to current transition>"
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

- time_gap_seconds can be approximate; it only needs to reflect that this is a **short** within-episode gap (e.g., 120 for 2 minutes, 900 for 15 minutes).
- task_reminders may contain 0, 1, or multiple entries depending on how many good within-episode pending-task opportunities exist in this batch.

⸻

### **Failure Case Format**

If you find no good episodic task reminder opportunities in the current batch (no clear "unfinished task + user moving on" cases), output:

{
  "service_main_type": "Long-Term Proactive Service",
  "service_sub_type": "Episodic Task Reminder Proactive Service",
  "task_reminders": [],
  "note": "No within-episode pending tasks with clear transition-away behavior were detected in this batch, given the current annotations."
}
"""