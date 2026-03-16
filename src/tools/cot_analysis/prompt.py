# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Prompt construction for CoT consistency analysis.

Builds the multimodal evaluation prompt sent to Gemini.
Same 2-dimension rubric as the alpamayo gemini_inference.py:
  1. CoT Coherence  (1-5)
  2. CoT-Output Alignment (1-5)
"""


def build_prompt(cot_text: str, traj_features: dict) -> str:
    """Build the CoT consistency evaluation prompt.

    Args:
        cot_text: The chain-of-thought reasoning text from the driver.
        traj_features: Dict from compute_trajectory_features().

    Returns:
        Formatted prompt string for Gemini.
    """
    markdown_kv = traj_features.get("markdown_kv", "No trajectory data available.")
    stats = traj_features.get("summary_stats", {})

    stats_text = (
        f"Duration: {stats.get('duration_s', '?')}s, "
        f"Path length: {stats.get('total_path_length_m', '?')}m, "
        f"Final position: ({stats.get('final_longitudinal_m', '?')}m forward, "
        f"{stats.get('final_lateral_m', '?')}m lateral), "
        f"Max lateral deviation: {stats.get('max_lateral_m', '?')}m / "
        f"{stats.get('min_lateral_m', '?')}m, "
        f"Speed: {stats.get('mean_speed_ms', '?')} avg / "
        f"{stats.get('max_speed_ms', '?')} max m/s"
    )

    prompt = f"""You are an expert evaluator for autonomous vehicle reasoning systems.
You are given the Chain-of-Thought (CoT) reasoning produced by a driving model, along with
the raw numerical trajectory it actually produced.

=== INPUT ===

**Chain-of-Thought Reasoning:**
"{cot_text}"

**Predicted Trajectory (ego-frame: +X = forward, +Y = left, sampled at ~1Hz):**
{markdown_kv}

**Summary:** {stats_text}

=== COORDINATE FRAME ===
- X axis: forward (longitudinal). Positive X = vehicle moves forward.
- Y axis: lateral. Positive Y = leftward from vehicle center. Negative Y = rightward.
- lat_vel: lateral velocity. Positive = moving left. Negative = moving right.
- accel: longitudinal acceleration. Positive = speeding up. Negative = braking.

=== EVALUATION CRITERIA ===
**Step 1: Scoring (1-5)**
Provide a holistic score (1-5) evaluating the following dimensions. Ensure to provide justification for score deductions.

**Dimension 1: CoT Coherence**
Is the reasoning internally logical? Does it reference a clear cause (perception) leading
to a clear effect (action)?
Rubric for dimension 1:
- *Score 5*: Perfectly logical. Clearly identifies a specific visual trigger and logically derives the appropriate action without any contradictions.
- *Score 4*: Mostly logical. The causal link is sound, but perception might be vague (e.g., "obstacle" instead of "pedestrian") or relies on generic templates.
- *Score 3*: Partial coherence. The reasoning is disjointed, relies heavily on generic statements without clear cause/effect, or has minor logical gaps.
- *Score 2*: Major logical flaws. The stated cause and effect strongly contradict each other in standard driving logic (e.g., "Light is red, so I accelerate").
- *Score 1*: Completely incoherent, fragmented, or fails to establish any causal relationship.

Example:
1)
  - CoT: "Nudge left to increase clearance from the parked vehicle on the right shoulder."
  - Justification: "The reasoning perfectly links a specific visual trigger (parked vehicle on the right) to a logical evasive action (nudge left)."
  - Score: 5
2)
  - CoT: "Decelerate because the other vehicles are stopped at the junction and the junction is empty"
  - Justification: "The reasoning links a valid cause (stopped vehicles) to a logical effect (decelerate), but suffers from partial coherence due to a direct contradiction in its scene description ('vehicles stopped at the junction' vs 'junction is empty')."
  - Score: 3
3)
  - CoT: "The light is red, so I accelerate."
  - Justification: "The reasoning establishes a clear cause (red light) and effect (acceleration), but the cause-effect relationship is fundamentally illogical in standard driving behavior."
  - Score: 2

**Dimension 2: CoT-Output Alignment**
Does the stated action match the trajectory?
- *Rule 1 (Lateral):* If CoT says "nudge" or "change lanes", `lat_vel` or `y` MUST show significant lateral movement. If CoT says "nudge left" but Y stays near 0 or goes negative, score 1-2.
- *Rule 2 (Distance/Speed):* If CoT says "keep distance" or "adapt speed", verify kinematic plausibility. You must look at the images to identify the lead vehicle, then look at `accel`. If `accel` is aggressively positive while approaching a visible lead vehicle, score 1-2. Smooth/zero or negative acceleration is consistent with keeping distance.
Rubric for dimension 2:
- **Score 5: Perfect alignment.** The trajectory perfectly executes the stated action in the CoT (correct direction and kinematic magnitude).
- **Score 4: High alignment.** The trajectory correctly executes the stated action, but with minor discrepancies in magnitude (e.g., CoT says "hard brake" but trajectory shows moderate deceleration).
- **Score 3: Partial alignment.** The trajectory loosely matches the intent, but there is a noticeable mismatch in execution (e.g., CoT says "hard brake" but trajectory shows only a slight deceleration, or "nudge" with very small lateral movement).
- **Score 2: Poor alignment.** The trajectory diverges significantly from the stated intent (e.g., CoT says "nudge left" but trajectory is perfectly straight).
- **Score 1: Complete contradiction.** The actual trajectory directly contradicts the stated CoT behavior (e.g., CoT says "yield" but trajectory accelerates heavily, or CoT says "steer left" but trajectory steers right).

Example:
1)
  - CoT: "Keep lane to continue driving since no critical agent needs attention."
  - Trajectory: "'y' stays near 0, 'accel' is near 0."
  - Justification: "The reasoning perfectly links a specific visual trigger (no critical agent) to a logical action (keep lane)."
  - Score: 5
2)
  - CoT: "Keep speed due to the lead vehicle maintaining a steady pace with a stable gap ahead."
  - Trajectory: Negative 'accel', 'y' increases.
  - Justification: "The CoT states 'Keep speed', but the trajectory shows significant deceleration (from a to b m/s) and a substantial left turn (c m lateral displacement) which are not mentioned."
  - Score: 2

Considerations:
- Some driving actions (e.g., "nudge") are subtle and may not be reflected in the trajectory when the long term trajectory is dominated by other factors such as following the road curvature. Be generous when scoring these actions.

=== OUTPUT FORMAT ===

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "cot_coherence": {{
    "justification": "<one sentence>",
    "score": <1-5>
  }},
  "cot_output_alignment": {{
    "justification": "<one sentence linking the text to specific trajectory numbers>",
    "score": <1-5>
  }},
  "overall_assessment": "<one sentence summary>"
}}"""
    return prompt
