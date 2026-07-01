SAFETY_PROMPT = """
### **Task Context**

You are analyzing MULTIPLE egocentric annotation files (each less than 10 minutes) from the EgoLife dataset for one contiguous batch of time (less than 2.5 hours).

Your job in this stage is signal mining and dialogue generation for a special Instant Proactive Service subtype:

Instant Safety Proactive Service  
(short: Instant Safety)

This service focuses on **acute physical hazards within a very short time window (≈0-10 seconds)**, where:
- The user is **currently performing** an action or is in a configuration that can **immediately cause bodily harm** (cut, burn, slip, collision, electric shock, etc.).
- The danger is **present right now or in the next few seconds** (no more than 10 seconds), not a long-term ergonomic or lifestyle issue.
- A short, timely warning from the assistant could realistically help the user **avoid injury in this moment**.

Typical examples:
    - The user's fingers are very close to a knife blade while cutting.
    - The user reaches over a boiling pot or an open flame with bare hands or loose clothing.
    - There is spilled liquid on the floor right where the user is stepping.
    - The user touches or approaches exposed electrical wiring or a wet powered device.
    - The user stands or walks very near a moving vehicle, trolley, or heavy object.

***You must:***
1. Scan the current batch to detect **instant safety hazards** grounded in the annotations.
2. For each hazard, select the **current moment** where:
   - the hazard is already present or about to occur within a few seconds, and
   - a warning would still be timely and helpful.
3. For each such moment, generate a short multi-turn dialogue where the assistant:
   - issues a clear, concise safety warning, and
   - offers concrete, practical suggestions to reduce the immediate risk.

There is **no historical memory JSON** for this service:
- You treat each batch independently.
- You do **not** track cross-day or long-horizon safety state.
- You only output dialogues for **this batch**, each grounded in **one current hazardous episode**.

You must NOT guess new hazards, invent timestamps, or hallucinate behaviors not supported by the annotations.

⸻

### **Authoritative Timestamps & Fine-Grained Evidence Priority**

Each fine-grained annotation entry has a time_window aligned to the original video, for example:
    - i_do_steps[].time_window
    - interactions_with_objects[].time_window
    - interactions_with_people[].time_window
    - speakers_say[].time_window

All time_window strings already have the format:

> DAY# HH:MM:SS-HH:MM:SS
  
When you output current_time_window, you MUST copy it only from i_do_steps[].time_window or speakers_say[].time_window.
You must NOT use interactions_with_objects, interactions_with_people, or interaction_records as the source of current_time_window.
Do NOT interpolate, merge, extend, shrink, or invent new time ranges.

**Fine-grained evidence priority**

When detecting Instant Safety opportunities, you must prioritize fine-grained entries:

- Primary sources to consider as candidate hazardous episodes:
  - i_do_steps
  - interactions_with_objects
  - interactions_with_people
  - speakers_say

Treat each fine-grained entry as a potential Instant Safety episode **only if its content clearly matches the safety definition** (immediate physical hazard within a few seconds).

This priority applies both to:

- which content you treat as hazardous episodes, and  
- which time_window you copy for those episodes.

⸻

### **Input Human Annotations**

You are given MULTIPLE JSON objects, each representing a <= 10-minute egocentric segment:
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
You can assume this coverage is at most about 2.5 hours.

⸻

### **Service Category**

- service_main_type: "Instant Proactive Service"
- service_sub_type: "Safety Proactive Service"

Note: this service operates on **instant, short-horizon physical hazards (≈0-10 seconds)** and is distinct from:

- Episodic Memory Recall (short-horizon episodic memory across ≤ 2.5 hours),
- Episodic Task Reminder (unfinished tasks inside one episode),
- Habit-Coaching or Long-Term Safety Habits (chronic risk, posture, lifestyle),
- Long-Horizon Memory-Link (cross-day or ≥2h links).

⸻

### **What counts as an Instant Safety signal?**

An Instant Safety opportunity is defined by **one current hazardous episode**, characterized by:

1. **Concrete hazardous condition**

   The annotations indicate that the user is interacting with, or very near to, a hazard such as:

   - open flame, very hot cookware, boiling liquids,
   - knife edges or sharp tools near hands/body,
   - spilled liquids on the floor, loose cables that could cause tripping,
   - exposed wires, wet or damaged electrical equipment,
   - unstable heavy objects that might fall,
   - nearby moving vehicles, bikes, trolleys, or machines.

2. **Immediate risk window**

   The risk is:

   - present **now** or in the **next few seconds**,
   - not yet mitigated (no evidence of moving away, switching off, etc.),
   - clearly more than a mild ergonomic or comfort issue.

The core question:  
> “If a careful assistant were watching this exact moment, would they reasonably say:  
>  *'Be careful right now, you might get hurt in the next few seconds.'*?”

If yes, it is a strong candidate for Instant Safety.

⸻

### **Mutual Exclusion & Priority vs. Other Services**

Use these rules to distinguish Safety from other proactive services:

1. Safety vs. Tool Use
   - Safety: the main concern is immediate injury risk (cut, burn, shock, collision) that could happen in the next few seconds.  
     Even if the user is also using a tool in a suboptimal way, label it Safety when physical danger is high and imminent.
   - Tool Use: the issue is mainly about **how** the user operates a tool (grip, angle, stability, posture), not about an acute bodily hazard.  
     Typical Tool Use cases are: holding a knife in an unstable way, using a screwdriver at an awkward angle, or standing too far/close for good leverage, but **no clear risk of being cut/burned/shocked in the next few seconds**.  
     A Tool Use message sounds like “hold the knife like this for better control” or “adjust your stance so the drill is more stable,” rather than “watch out, you might get hurt right now.”

2. Safety vs. Error-Recovery
   - Error-Recovery: the user chooses the wrong object/step/target/configuration **in a multi-step workflow**, and the main problem is that they must **roll back to an earlier step and redo correctly**, not that they are in immediate physical danger.  
     For example, the user does step 1 and step 2 of a procedure, realizes step 2 was wrong, and needs to go back to the state after step 1 and repeat step 2; or they pour a reagent into the wrong container but there is no acute hazard.
   - If the same workflow error also creates an acute physical danger (e.g., placing a flammable liquid beside an open flame, mis-wiring something that can shock them), then for this service it should be treated as **Safety**, not pure Error-Recovery.

3. Safety vs. Resource Reminder
   - Resource Reminder: the user is about to leave non-closed states (stove still on, door unlocked, device not shut down) but there is **no immediate serious hazard at this exact moment** (e.g., low heat with a stable pot and no overflow yet).
   - If the leftover state is already causing immediate serious danger (e.g., an unattended pot already boiling over, gas flame under an almost dry pan), treat it as **Safety** here.

4. Safety vs. Habit-Coaching / Long-Term
   - Posture, screen distance, repetitive strain, or lifestyle risk **without a specific acute moment of injury** belong to Habit-Coaching or Long-Term services, not Instant Safety.

⸻

### **Time-Window Rules (strict)**

For every Instant Safety dialogue you output, you must define a single:

- **current_time_window** representing the hazardous episode and the moment when the assistant speaks.

#### Rules:

1. **Current time window (current_time_window)**
   - Prefer to copy from one fine-grained entry's time_window:
     - i_do_steps[].time_window, or
     - interactions_with_objects[].time_window, or
     - interactions_with_people[].time_window, or
     - speakers_say[].time_window.
   - Do NOT merge or fabricate new ranges.

2. **Instant horizon**
   - current_time_window should correspond to the **short segment (≈0-10 seconds)** where:
     - the hazardous configuration arises, and
     - a warning would still be in time to prevent harm.

The assistant's first proactive utterance is interpreted as happening **during or immediately after** this current_time_window.

⸻

### **Instant Safety Detection Objective**

For the current batch (no external history):

1. Scan all segments and interaction_records to find **candidate hazardous episodes** where:
   - fine-grained annotations describe risky objects, body positions, or environmental states, and
   - the risk could cause physical injury within the next few seconds.

2. For each candidate, verify:
   - There is a **specific hazard** (e.g., knife near fingers, boiling water near hands).
   - The risk is **immediate** (not vague or long-term, no more than 10 seconds).
   - There is no evidence that the user has already moved away or fully mitigated the risk.

3. For each valid hazardous episode:
   - Choose the most fine-grained time_window that best captures the critical 0-10 second window, and this time_window MUST come from i_do_steps[].time_window or speakers_say[].time_window in the annotations.
   - Record the local scene context and a short explanation of why this is dangerous now.

4. For each such episode, generate:
   - One structured Instant Safety entry with:
     - current_segment_id, current_time_window, supporting_source,
     - a neutral scene description,
     - hazard_key, hazard_summary, risk_type, potential_consequence,
     - occurrence_confidence.
   - One multi-turn dialogue where the assistant issues a timely safety warning.

You may output multiple safety events in one batch, but:

- Do not duplicate events for trivially overlapping time_windows describing the same hazard.
- If multiple fine-grained annotations describe the same continuous hazard, you may choose **one representative** time_window.

⸻

### **Dialogue Generation Objective**

For each Instant Safety episode, you must generate a short multi-turn dialogue that:

    1. **Grounds in the specific hazard**

    - The assistant should explicitly or implicitly mention:
        - the risky object or surface,
        - the relevant body part or position,
        - the immediate nature of the risk.
    - Examples:
        - “Watch your fingers; the knife blade is very close.”
        - “Be careful, the pot on your left is boiling and might splash.”
        - “There's liquid on the floor right where you're walking.”

    - Do **not** mention “video”, “annotations”, “models”, or internal logging.

    2. **Keeps focus on the present moment**

    - Make it clear that the warning is about what is happening **right now**, e.g.:
        - “As you reach across the stove…”
        - “Right in front of your feet there's a wet spot.”
        - “While you're plugging this in, the socket looks exposed.”

    3. **Offers concrete, practical suggestions**

    - Suggest one or two simple actions to reduce risk, such as:
        - adjusting grip or hand position,
        - stepping around a spill,
        - turning down the heat,
        - moving flammable items away,
        - unplugging or avoiding a damaged cable.

    4. **Tone and emotional value**

    - Calm, supportive, non-panicking.
    - Emphasize care and prevention, not blame:
        - “just so you don't get hurt,”
        - “to stay safe while you're doing this.”
    - Avoid scolding language (“you're being careless”); instead:
        - “If you slow down a bit and move your hand back…”
        - “You might want to wipe that up before you walk through.”

    5. **Dialogue structure**

    Each proactive_dialogue should contain at least 2 turns, preferably 3-4:

        1. **Assistant (Turn 1)**  
        - Proactively issues a safety warning,
        - names or hints at the hazard,
        - and suggests a simple mitigating action.

        2. **User (Turn 2)**  
        - Natural reply (≥ 12 English words).
        - May:
            - acknowledge and adjust,
            - explain they already noticed,
            - ask for a specific suggestion,
            - or briefly decline (“I'm okay, but thanks for the reminder”).

        3. **Assistant (optional Turn 3)**  
        - Acknowledge the user's response,
        - offer a refined suggestion or reassurance,
        - or back off politely while keeping safety in mind.

        4. **User (optional Turn 4)**  
        - Short acknowledgement or final confirmation.

        **Diversity**
        Across different safety dialogues in the same output:
            - Vary how you phrase the warning:
            - “Be careful…”
            - “Watch out…”
            - “Just a heads-up…”
            - Vary the help framing:
            - “Maybe move your hand a little further back.”
            - “You could step around that spill.”
            - “It might be safer to turn the heat down a bit.”

Not every user must respond enthusiastically; some may give a brief answer or lightly downplay the risk.

⸻

### **Output Format**

Your output MUST be ONE JSON object:

{
  "service_main_type": "Instant Proactive Service",
  "service_sub_type": "Safety Proactive Service",

  "safety_events": [
    {
      "event_id": "safety_001",

      "current_segment_id": "<segment id where the assistant speaks>",
      "current_time_window": "DAY# HH:MM:SS-HH:MM:SS",
      "supporting_source": "i_do_steps | speakers_say",

      "scene_description": "<short neutral description of what the user is doing around this moment>",
      "trigger_reason": "<why this moment is an Instant Safety opportunity, grounded in the annotations>",

      "hazard_key": "<abstract description of the hazard, e.g., 'knife near fingers', 'spilled liquid underfoot'>",
      "hazard_summary": "<short explanation of what is dangerous and why now>",
      "risk_type": "cut | burn | slip | collision | electric_shock | other",
      "risk_immediacy": "immediate",
      "potential_consequence": "<optional: what could happen (scald, cut, fall, etc.)>",

      "occurrence_confidence": 0.0,

      "proactive_dialogue": [
        {
          "role": "assistant",
          "utterance": "<assistant issues an instant safety warning, grounded in the hazard and context>"
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
- safety_events may contain 0, 1, or multiple entries depending on how many Instant Safety opportunities exist in this batch.
- You may approximate the severity in potential_consequence, but it must remain consistent with the annotations.

⸻

### **Failure Case Format**

If you find no Instant Safety opportunities in the current batch:

{
  "service_main_type": "Instant Proactive Service",
  "service_sub_type": "Safety Proactive Service",
  "safety_events": [],
  "note": "No Instant Safety Proactive Service opportunities were detected in the current batch, given the current annotations."
}
"""