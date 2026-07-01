"""
Proactive service prompts for EgoLife benchmark.
Reference: GST_VIDEORAG/VideoRAG-algorithm/videorag/ego_prompt_.py
"""

# ============================================================================
# Proactive Service Taxonomy Definition (shared across all tasks)
# ============================================================================

SERVICE_TAXONOMY = """
------------------------------------------------------------
Proactive Service Taxonomy
------------------------------------------------------------

A) Instant Proactive Services (<= 10 seconds, current moment only)

1. Safety
   Immediate physical danger visible now.
   Examples: hand near open flame, unstable position near hazard,
   object moving uncontrollably or could fall immediately.

2. Tool Use
   Tool or device being used in an unsafe or unstable way right now.
   Examples: tool held at unstable angle, guard/cover missing during use,
   powered device running while handled improperly.

B) Short-Term Proactive Services (10 seconds to ~10 minutes, same session)

1. Error-Recovery
   A clearly incorrect action just occurred and must be corrected.
   Examples: wrong component attached, device in wrong mode,
   object inserted into incorrect location.

2. Next-Step Guidance
   An explicit pause, idle state, or visible hesitation AFTER a completed substep.
   Examples: user finishes preparing materials but does not begin next step,
   tool put down after step with no follow-up, user looks around after substep.

3. Resource Reminder
   User moves on while leaving an unresolved state behind.
   Examples: device remains powered on after use, container/door left open,
   materials or tools left unsecured after context shift.

C) Episodic Proactive Services (~10 minutes to ~2.5 hours, same day)

1. Episodic Task Reminder
   An earlier task from the same day appears unfinished,
   and the current moment shows disengagement or context switching.
   Examples: food preparation started then abandoned, form/setup begun then left,
   items prepared for an activity that did not proceed.

2. Episodic Memory Recall
   Current moment suggests the user needs something handled earlier.
   Examples: user searches multiple locations, pauses as if checking for missing item,
   reaches toward usual location where object is not present.

D) Long-Term Proactive Services (>= 2.5 hours, cross-session or multi-day)

1. Long-Horizon Memory-Link
   An action/decision made hours or days earlier directly affects now.
   Examples: prior setup enables/constrains current task,
   object placed earlier is now required, prior configuration affects device.

2. Routine Optimization
   Stable repeated routines observed across sessions that could be streamlined.
   Examples: same multi-step setup repeated daily, environment configs repeatedly adjusted,
   daily habits follow consistent pattern, fixed but inefficient task order.

3. Personal Progress Feedback
   Repeated execution of same activity shows observable improvement over time.
   Examples: physical actions faster/more precise, complex movements smoother,
   repeated tasks show reduced hesitation or correction.

4. Habit-Coaching
   Unhealthy or suboptimal behaviors accumulate over extended time.
   Examples: prolonged sitting (>40 min), frequent short-interval phone checks,
   extended time without hydration (>2 hours), late working hours, irregular meals.
"""

# ============================================================================
# Main proactive service detection prompt (for each 30s video segment)
# ============================================================================

PROACTIVE_DETECTION_PROMPT = """You are a proactive service detection assistant for egocentric video.

You will be given {num_frames} uniformly sampled frames from a video segment of ~{duration_seconds} seconds.
The segment is from {person_id}, {day_id}, time window: {time_window}.
Frame timestamps (in seconds from segment start): {frame_timestamps}

Your task: analyze the visual content and determine if any proactive service should be triggered.

""" + SERVICE_TAXONOMY + """

------------------------------------------------------------
Instructions
------------------------------------------------------------

1. Examine the frames and decide whether any proactive service is warranted.
2. For each detected service, provide your reasoning process and specify the
   time span (in seconds relative to segment start) where the service applies.
   The time span should typically be <= 5 seconds.
3. If NO service is warranted, output an empty services list.
4. Be conservative: only trigger a service when visual evidence is clear.
5. Do NOT infer intentions, emotions, or plans — only use what is visible.

------------------------------------------------------------
Output Format (STRICT JSON, nothing else)
------------------------------------------------------------

{{
  "services": [
    {{
      "service_main_type": "Instant | Short-Term | Episodic | Long-Term",
      "service_sub_type": "<one of: Safety, Tool Use, Error-Recovery, Next-Step Guidance, Resource Reminder, Episodic Task Reminder, Episodic Memory Recall, Long-Horizon Memory-Link, Routine Optimization, Personal Progress Feedback, Habit-Coaching>",
      "time_span": [<start_seconds>, <end_seconds>],
      "confidence": "high | medium",
      "reasoning": "<step-by-step reasoning: what you see in the frames and why this service is needed>",
      "trigger_evidence": "<concise factual evidence from the frames>",
      "user_prompt": "<short supportive message to the user, 1-2 sentences>"
    }}
  ]
}}

If no service is needed, output exactly:
{{
  "services": []
}}
"""

# ============================================================================
# Simple caption prompt (for generating frame descriptions before detection)
# ============================================================================

# ============================================================================
# Dataset-specific taxonomy: HoloAssist (Instant + Short-Term only)
# ============================================================================

HOLOASSIST_SERVICE_TAXONOMY = """
------------------------------------------------------------
Proactive Service Taxonomy
------------------------------------------------------------

A) Instant Proactive Services (<= 10 seconds, current moment only)

1. Safety
   Immediate physical danger visible now.
   Examples: hand near open flame, unstable position near hazard,
   object moving uncontrollably or could fall immediately.

2. Tool Use
   Tool or device being used in an unsafe or unstable way right now.
   Examples: tool held at unstable angle, guard/cover missing during use,
   powered device running while handled improperly.

B) Short-Term Proactive Services (10 seconds to ~10 minutes, same session)

1. Error-Recovery
   A clearly incorrect action just occurred and must be corrected.
   Examples: wrong component attached, device in wrong mode,
   object inserted into incorrect location.

2. Next-Step Guidance
   An explicit pause, idle state, or visible hesitation AFTER a completed substep.
   Examples: user finishes preparing materials but does not begin next step,
   tool put down after step with no follow-up, user looks around after substep.

3. Resource Reminder
   User moves on while leaving an unresolved state behind.
   Examples: device remains powered on after use, container/door left open,
   materials or tools left unsecured after context shift.
"""

HOLOASSIST_DETECTION_PROMPT = """You are a proactive service detection assistant for egocentric video.

You will be given {num_frames} uniformly sampled frames from a video segment of ~{duration_seconds} seconds.
The segment is from {person_id}, {day_id}, time window: {time_window}.
Frame timestamps (in seconds from segment start): {frame_timestamps}

Your task: analyze the visual content and determine if any proactive service should be triggered.

""" + HOLOASSIST_SERVICE_TAXONOMY + """

------------------------------------------------------------
Instructions
------------------------------------------------------------

1. Examine the frames and decide whether any proactive service is warranted.
2. For each detected service, provide your reasoning process and specify the
   time span (in seconds relative to segment start) where the service applies.
   The time span should typically be <= 5 seconds.
3. If NO service is warranted, output an empty services list.
4. Be conservative: only trigger a service when visual evidence is clear.
5. Do NOT infer intentions, emotions, or plans — only use what is visible.

------------------------------------------------------------
Output Format (STRICT JSON, nothing else)
------------------------------------------------------------

{{
  "services": [
    {{
      "service_main_type": "Instant | Short-Term",
      "service_sub_type": "<one of: Safety, Tool Use, Error-Recovery, Next-Step Guidance, Resource Reminder>",
      "time_span": [<start_seconds>, <end_seconds>],
      "confidence": "high | medium",
      "reasoning": "<step-by-step reasoning: what you see in the frames and why this service is needed>",
      "trigger_evidence": "<concise factual evidence from the frames>",
      "user_prompt": "<short supportive message to the user, 1-2 sentences>"
    }}
  ]
}}

If no service is needed, output exactly:
{{
  "services": []
}}
"""


# ============================================================================
# Dataset-specific taxonomy: CaptionCook4D (Instant + Short-Term + Episodic)
# ============================================================================

CAPTIONCOOK4D_SERVICE_TAXONOMY = """
------------------------------------------------------------
Proactive Service Taxonomy
------------------------------------------------------------

A) Instant Proactive Services (<= 10 seconds, current moment only)

1. Safety
   Immediate physical danger visible now.
   Examples: hand near open flame, unstable position near hazard,
   object moving uncontrollably or could fall immediately.

2. Tool Use
   Tool or device being used in an unsafe or unstable way right now.
   Examples: tool held at unstable angle, guard/cover missing during use,
   powered device running while handled improperly.

B) Short-Term Proactive Services (10 seconds to ~10 minutes, same session)

1. Error-Recovery
   A clearly incorrect action just occurred and must be corrected.
   Examples: wrong component attached, device in wrong mode,
   object inserted into incorrect location, wrong ingredient used.

2. Next-Step Guidance
   An explicit pause, idle state, or visible hesitation AFTER a completed substep.
   Examples: user finishes preparing materials but does not begin next step,
   tool put down after step with no follow-up, user looks around after substep.

3. Resource Reminder
   User moves on while leaving an unresolved state behind.
   Examples: device remains powered on after use, container/door left open,
   materials or tools left unsecured after context shift.

C) Episodic Proactive Services (~10 minutes to ~2.5 hours, same session)

1. Episodic Task Reminder
   An earlier task from the same session appears unfinished,
   and the current moment shows disengagement or context switching.
   Examples: food preparation started then abandoned, form/setup begun then left,
   items prepared for an activity that did not proceed.

2. Episodic Memory Recall
   Current moment suggests the user needs something handled earlier.
   Examples: user searches multiple locations, pauses as if checking for missing item,
   reaches toward usual location where object is not present.
"""

CAPTIONCOOK4D_DETECTION_PROMPT = """You are a proactive service detection assistant for egocentric video.

You will be given {num_frames} uniformly sampled frames from a video segment of ~{duration_seconds} seconds.
The segment is from {person_id}, {day_id}, time window: {time_window}.
Frame timestamps (in seconds from segment start): {frame_timestamps}

Your task: analyze the visual content and determine if any proactive service should be triggered.

""" + CAPTIONCOOK4D_SERVICE_TAXONOMY + """

------------------------------------------------------------
Instructions
------------------------------------------------------------

1. Examine the frames and decide whether any proactive service is warranted.
2. For each detected service, provide your reasoning process and specify the
   time span (in seconds relative to segment start) where the service applies.
   The time span should typically be <= 5 seconds.
3. If NO service is warranted, output an empty services list.
4. Be conservative: only trigger a service when visual evidence is clear.
5. Do NOT infer intentions, emotions, or plans — only use what is visible.

------------------------------------------------------------
Output Format (STRICT JSON, nothing else)
------------------------------------------------------------

{{
  "services": [
    {{
      "service_main_type": "Instant | Short-Term | Episodic",
      "service_sub_type": "<one of: Safety, Tool Use, Error-Recovery, Next-Step Guidance, Resource Reminder, Episodic Task Reminder, Episodic Memory Recall>",
      "time_span": [<start_seconds>, <end_seconds>],
      "confidence": "high | medium",
      "reasoning": "<step-by-step reasoning: what you see in the frames and why this service is needed>",
      "trigger_evidence": "<concise factual evidence from the frames>",
      "user_prompt": "<short supportive message to the user, 1-2 sentences>"
    }}
  ]
}}

If no service is needed, output exactly:
{{
  "services": []
}}
"""


# ============================================================================
# Simple caption prompt (for generating frame descriptions before detection)
# ============================================================================

FRAME_CAPTION_PROMPT = """You are an egocentric episodic frame recorder.

You will be given {num_frames} uniformly sampled frames from a ~30-second egocentric video segment.
The segment is from {person_id}, {day_id}, time window: {time_window}.

For each frame, provide a concise first-person description of what is visually observable.
Then provide ONE global caption summarizing the full segment.

Focus on:
- Concrete physical actions (use action verbs: turn, look, reach, pick up, place, hold, press, open, close)
- Object interactions and states (on/off, open/closed, held/placed)
- Spatial relations (on the desk, beside my hand, in front of me)
- Unresolved states, errors, unsafe conditions, repetitions
- Presence of other people and their actions (no appearance/identity details)

Output a valid JSON object:
{{
  "caption": "<30-second global first-person caption, 2-4 sentences>",
  "frames": {{
    "0": "<first frame description, 1-2 sentences>",
    "1": "<second frame description>",
    ...
  }}
}}
"""
