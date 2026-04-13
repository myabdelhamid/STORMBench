"""
StormVLM — Baseline vs Optimized Evaluation Node
===================================================
Master's Thesis · GIU Berlin
Author: Marwan Elsayed

Fair, automated scoring system that compares the baseline (pre-optimization)
pipeline against the optimized (post-optimization) pipeline across all 3 nodes.

Scoring uses YAML ground truth + physics-based heuristics.
Both pipelines receive the same input data — only the prompt engineering
and post-processing differ.

Metrics (total 30 points):
  Perception (PA)  — 10 points
  Prediction (Pk)  — 10 points
  Planning   (Pl)  — 10 points
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from perception_node import AnnotationLoader, PerceptionResult, _VIEW_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="[StormVLM %(levelname)s] %(message)s",
)
log = logging.getLogger("stormvlm.evaluation")


# ═══════════════════════════════════════════════════════════════════════════
# Score Containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MetricScore:
    """A single metric's score."""
    name: str
    score: float
    max_score: float
    detail: str = ""

    @property
    def pct(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0


@dataclass
class NodeScore:
    """Aggregated score for one node (Perception, Prediction, or Planning)."""
    node_name: str
    metrics: list[MetricScore] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(m.score for m in self.metrics)

    @property
    def max_total(self) -> float:
        return sum(m.max_score for m in self.metrics)


@dataclass
class PipelineScore:
    """Full pipeline evaluation result."""
    pipeline_name: str
    perception: NodeScore = field(default_factory=lambda: NodeScore("Perception"))
    prediction: NodeScore = field(default_factory=lambda: NodeScore("Prediction"))
    planning: NodeScore = field(default_factory=lambda: NodeScore("Planning"))

    @property
    def total(self) -> float:
        return self.perception.total + self.prediction.total + self.planning.total

    @property
    def max_total(self) -> float:
        return self.perception.max_total + self.prediction.max_total + self.planning.max_total


# ═══════════════════════════════════════════════════════════════════════════
# Perception Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class PerceptionEvaluator:
    """
    Scores perception output against YAML ground truth.

    Metrics (10 points total):
      Object Coverage      (3) — Are the closest objects mentioned?
      Visibility Assessment(2) — Correct visibility given weather/distance?
      Radar Integration    (2) — Does it reference radar data?
      Location Awareness   (1) — Does it describe object positions?
      Output Actionability (2) — Can the next node consume this output?
    """

    @staticmethod
    def evaluate(
        perception_result: PerceptionResult,
        annotations: dict,
        radar_anchors: list[dict],
    ) -> NodeScore:
        """Score the perception output."""
        node = NodeScore(node_name="Perception (PA)")

        # Gather all reasoning text across views
        all_reasoning = ""
        for vp in perception_result.views.values():
            all_reasoning += " " + vp.reasoning

        all_reasoning_lower = all_reasoning.lower()

        # Get ground truth objects (top 3 closest per view, matching what the prompt uses)
        all_objects = annotations.get("vehicles", []) + annotations.get("walkers", [])
        sorted_objects = sorted(all_objects, key=lambda d: d.get("dist", float("inf")))
        close_objects = sorted_objects[:12]  # top 3 per view × 4 views

        weather = annotations.get("weather_type", "fd").lower()

        # ── Metric 1: Object Coverage (3 points) ─────────────────────────
        # Check if the output mentions detected objects
        objects_mentioned = 0
        total_gt_objects = min(len(close_objects), 12)

        for obj in close_objects:
            cls_name = obj.get("class", "object").lower()
            dist = obj.get("dist", 0.0)

            # Check various ways the object might be mentioned
            mentioned = False
            # Check for class name mention
            if cls_name in all_reasoning_lower:
                mentioned = True
            # Check for tagged format <class_N>
            if f"<{cls_name}" in all_reasoning_lower:
                mentioned = True
            # Check for distance mention (within 5m tolerance)
            dist_str = f"{dist:.1f}"
            if dist_str in all_reasoning:
                mentioned = True
            # Check approximate distance
            dist_int = str(int(round(dist)))
            if f"{dist_int}m" in all_reasoning_lower or f"{dist_int} m" in all_reasoning_lower:
                mentioned = True

            if mentioned:
                objects_mentioned += 1

        if total_gt_objects > 0:
            coverage_ratio = objects_mentioned / total_gt_objects
        else:
            coverage_ratio = 1.0  # No objects to detect = full score

        coverage_score = round(coverage_ratio * 3.0, 1)
        node.metrics.append(MetricScore(
            name="Object Coverage",
            score=min(coverage_score, 3.0),
            max_score=3.0,
            detail=f"{objects_mentioned}/{total_gt_objects} GT objects referenced",
        ))

        # ── Metric 2: Visibility Assessment (2 points) ───────────────────
        # Close objects (<15m) should be marked visible even in fog
        # Far objects (>30m) in fog should be marked not visible or uncertain
        visibility_score = 0.0
        visibility_checks = 0
        visibility_correct = 0

        for obj in close_objects[:6]:  # Check the 6 closest
            dist = obj.get("dist", 0.0)
            cls_name = obj.get("class", "object").lower()
            visibility_checks += 1

            if dist < 15.0:
                # Close objects should be visible regardless of weather
                if any(kw in all_reasoning_lower for kw in [
                    "visible", "can see", "clearly", "visible_in_camera\": true",
                    "visible_in_camera\":true", '"visible_in_camera": true'
                ]):
                    visibility_correct += 1
                elif "not visible" not in all_reasoning_lower and "hidden" not in all_reasoning_lower:
                    # Didn't explicitly say not visible = partial credit
                    visibility_correct += 0.5
            elif dist > 30.0 and "f" in weather:  # foggy
                # Far objects in fog should be uncertain/not visible
                if any(kw in all_reasoning_lower for kw in [
                    "not visible", "hidden", "fog", "obscured",
                    "visible_in_camera\": false", '"visible_in_camera": false'
                ]):
                    visibility_correct += 1
                else:
                    visibility_correct += 0.5  # Partial if didn't say visible explicitly

        if visibility_checks > 0:
            visibility_score = (visibility_correct / visibility_checks) * 2.0
        else:
            visibility_score = 2.0

        node.metrics.append(MetricScore(
            name="Visibility Assessment",
            score=round(min(visibility_score, 2.0), 1),
            max_score=2.0,
            detail=f"{visibility_correct:.1f}/{visibility_checks} correct assessments",
        ))

        # ── Metric 3: Radar Integration (2 points) ───────────────────────
        radar_score = 0.0
        has_radar = len(radar_anchors) > 0

        if has_radar:
            # Check if radar data is referenced at all
            radar_keywords = ["radar", "anchor", "confirmed", "has_radar",
                              "has_radar_anchor", "radar contact", "radar data"]
            radar_mentioned = any(kw in all_reasoning_lower for kw in radar_keywords)

            if radar_mentioned:
                radar_score = 1.0

                # Bonus: check if radar data influenced categorization
                cat_keywords = ["confirmed", "fogged", "ghost", "radar-confirmed",
                                "has_radar_anchor\": true", '"has_radar_anchor": true']
                if any(kw in all_reasoning_lower for kw in cat_keywords):
                    radar_score = 2.0
            else:
                radar_score = 0.0
        else:
            # No radar data available — not penalized
            radar_score = 1.0  # Neutral score

        node.metrics.append(MetricScore(
            name="Radar Integration",
            score=round(min(radar_score, 2.0), 1),
            max_score=2.0,
            detail=f"Radar {'referenced' if radar_score >= 1 else 'NOT referenced'} "
                   f"({'used for categorization' if radar_score >= 2 else 'mentioned only' if radar_score >= 1 else 'absent'})",
        ))

        # ── Metric 4: Location Awareness (1 point) ──────────────────────
        location_keywords = ["on road", "on sidewalk", "off road", "on the road",
                             "sidewalk", "roadside", "location_in_scene",
                             "on the sidewalk", "off the road"]
        location_mentioned = any(kw in all_reasoning_lower for kw in location_keywords)
        location_score = 1.0 if location_mentioned else 0.0

        node.metrics.append(MetricScore(
            name="Location Awareness",
            score=location_score,
            max_score=1.0,
            detail="Location described" if location_mentioned else "No location info",
        ))

        # ── Metric 5: Output Actionability (2 points) ────────────────────
        # Can the next node (Prediction) reliably parse and use this output?
        actionability_score = 0.0

        # Check if output is valid JSON (strongly actionable)
        is_json = False
        for vp in perception_result.views.values():
            text = vp.reasoning.strip()
            # Try JSON extraction
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    if "categorization" in data:
                        is_json = True
                        break
                except json.JSONDecodeError:
                    pass
            # Try raw JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if "categorization" in data:
                        is_json = True
                        break
                except json.JSONDecodeError:
                    pass

        if is_json:
            actionability_score = 2.0  # Full: structured JSON with categorization
        elif objects_mentioned > 0:
            actionability_score = 1.0  # Partial: mentions objects but unstructured
        else:
            actionability_score = 0.0  # Not actionable

        node.metrics.append(MetricScore(
            name="Output Actionability",
            score=actionability_score,
            max_score=2.0,
            detail="Structured JSON" if is_json else (
                "Text mentions objects" if actionability_score >= 1 else "Not actionable"
            ),
        ))

        return node


# ═══════════════════════════════════════════════════════════════════════════
# Prediction Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class PredictionEvaluator:
    """
    Scores prediction output against physics-based expectations.

    Metrics (10 points total):
      Risk Assessment     (3) — Physically plausible risk levels?
      Motion Prediction   (3) — Consistent with kinematics?
      Object Completeness (2) — All perception objects carried through?
      Reasoning Quality   (2) — Reasoning explains the risk level?
    """

    @staticmethod
    def evaluate(
        prediction_output: str,
        perception_result: PerceptionResult,
        annotations: dict,
    ) -> NodeScore:
        """Score the prediction output."""
        node = NodeScore(node_name="Prediction (Pk)")

        prediction_lower = prediction_output.lower()
        ego_speed = annotations.get("ego_speed", 0.0)

        # Gather all objects from perception
        all_objects = []
        for vp in perception_result.views.values():
            for d in vp.detections:
                all_objects.append({**d, "view": vp.view_name})

        sorted_objects = sorted(all_objects, key=lambda d: d.get("dist", float("inf")))
        relevant_objects = sorted_objects[:12]

        # Build physics-based expectations
        expectations = []
        for obj in relevant_objects:
            dist = obj.get("dist", 0.0)
            speed = obj.get("speed", 0.0)
            speed_xyz = obj.get("speed_x_y_z", [0.0, 0.0, 0.0])
            view = obj.get("view", "Front")
            vx = speed_xyz[0] if len(speed_xyz) > 0 else 0.0
            v_rel = vx - ego_speed

            # Determine expected risk
            if view == "Front":
                if dist < 15.0 and speed < 0.5:
                    expected_risk = "HIGH"
                elif dist < 15.0:
                    expected_risk = "MODERATE-HIGH"
                elif dist < 30.0 and v_rel < 0:
                    expected_risk = "MODERATE"
                elif v_rel > 0:  # moving away
                    expected_risk = "LOW-NONE"
                else:
                    expected_risk = "LOW"
            elif view == "Back":
                expected_risk = "LOW-NONE"  # Behind us
            else:
                expected_risk = "LOW"  # Sides

            expectations.append({
                "class": obj.get("class", "object"),
                "dist": dist,
                "speed": speed,
                "view": view,
                "v_rel": v_rel,
                "expected_risk": expected_risk,
            })

        # ── Metric 1: Risk Assessment (3 points) ─────────────────────────
        risk_score = 0.0
        risk_checks = 0
        risk_correct = 0.0

        # Try to parse JSON predictions
        pred_objects = PredictionEvaluator._extract_prediction_objects(prediction_output)

        if pred_objects:
            # JSON-based evaluation (more precise)
            for pred in pred_objects:
                risk = pred.get("risk_level", "").upper()
                # Match to expected
                for exp in expectations:
                    if exp["class"].lower() in pred.get("id", "").lower():
                        risk_checks += 1
                        expected = exp["expected_risk"]
                        if PredictionEvaluator._risk_matches(risk, expected):
                            risk_correct += 1.0
                        elif PredictionEvaluator._risk_close(risk, expected):
                            risk_correct += 0.5
                        break
        else:
            # Text-based evaluation (less precise, heuristic)
            for exp in expectations:
                cls_name = exp["class"].lower()
                if cls_name in prediction_lower:
                    risk_checks += 1
                    # Check if risk-related keywords match expectation
                    expected = exp["expected_risk"]
                    if "HIGH" in expected:
                        if any(kw in prediction_lower for kw in ["dangerous", "high risk", "imminent", "critical", "emergency"]):
                            risk_correct += 1.0
                        elif any(kw in prediction_lower for kw in ["risk", "caution", "careful"]):
                            risk_correct += 0.5
                    elif "LOW" in expected or "NONE" in expected:
                        if any(kw in prediction_lower for kw in ["low risk", "no risk", "safe", "no threat", "no danger"]):
                            risk_correct += 1.0
                        elif not any(kw in prediction_lower for kw in ["dangerous", "high risk", "critical"]):
                            risk_correct += 0.5

        if risk_checks > 0:
            risk_score = (risk_correct / risk_checks) * 3.0
        elif len(relevant_objects) == 0:
            risk_score = 3.0  # No objects to assess

        node.metrics.append(MetricScore(
            name="Risk Assessment",
            score=round(min(risk_score, 3.0), 1),
            max_score=3.0,
            detail=f"{risk_correct:.1f}/{risk_checks} risk levels physically plausible",
        ))

        # ── Metric 2: Motion Prediction (3 points) ───────────────────────
        motion_score = 0.0
        motion_checks = 0
        motion_correct = 0.0

        if pred_objects:
            for pred in pred_objects:
                motion = pred.get("future_motion", "").lower()
                for exp in expectations:
                    if exp["class"].lower() in pred.get("id", "").lower():
                        motion_checks += 1
                        v_rel = exp["v_rel"]
                        if v_rel > 0.5:  # Moving away
                            if any(kw in motion for kw in ["away", "diverging", "receding", "moving away"]):
                                motion_correct += 1.0
                            elif "stationary" not in motion:
                                motion_correct += 0.3
                        elif abs(v_rel) < 0.5 and exp["speed"] < 0.5:  # Stationary
                            if any(kw in motion for kw in ["stationary", "stopped", "still", "remain"]):
                                motion_correct += 1.0
                            elif "approaching" not in motion:
                                motion_correct += 0.3
                        elif v_rel < -0.5:  # Approaching
                            if any(kw in motion for kw in ["approach", "closing", "toward", "coming"]):
                                motion_correct += 1.0
                            elif "away" not in motion:
                                motion_correct += 0.3
                        break
        else:
            # Text-based heuristic
            for exp in expectations:
                cls_name = exp["class"].lower()
                if cls_name in prediction_lower:
                    motion_checks += 1
                    v_rel = exp["v_rel"]
                    speed = exp["speed"]
                    if speed < 0.5 and any(kw in prediction_lower for kw in ["stop", "stationary", "still", "parked"]):
                        motion_correct += 1.0
                    elif v_rel > 0.5 and any(kw in prediction_lower for kw in ["away", "leaving", "receding"]):
                        motion_correct += 1.0
                    elif v_rel < -0.5 and any(kw in prediction_lower for kw in ["approach", "toward", "coming", "closing"]):
                        motion_correct += 1.0
                    else:
                        motion_correct += 0.3  # Mentioned but motion unclear

        if motion_checks > 0:
            motion_score = (motion_correct / motion_checks) * 3.0
        elif len(relevant_objects) == 0:
            motion_score = 3.0

        node.metrics.append(MetricScore(
            name="Motion Prediction",
            score=round(min(motion_score, 3.0), 1),
            max_score=3.0,
            detail=f"{motion_correct:.1f}/{motion_checks} motion predictions plausible",
        ))

        # ── Metric 3: Object Completeness (2 points) ─────────────────────
        # How many perception-stage objects are carried through to prediction?
        objects_in_prediction = 0
        for obj in relevant_objects:
            cls_name = obj.get("class", "object").lower()
            if cls_name in prediction_lower:
                objects_in_prediction += 1

        total_expected = len(relevant_objects) if relevant_objects else 1
        completeness_ratio = min(objects_in_prediction / total_expected, 1.0)
        completeness_score = completeness_ratio * 2.0

        node.metrics.append(MetricScore(
            name="Object Completeness",
            score=round(min(completeness_score, 2.0), 1),
            max_score=2.0,
            detail=f"{objects_in_prediction}/{len(relevant_objects)} objects carried through",
        ))

        # ── Metric 4: Reasoning Quality (2 points) ───────────────────────
        # Does the prediction explain WHY risk levels were assigned?
        reasoning_score = 0.0

        # Check for causal reasoning keywords
        causal_keywords = ["because", "since", "due to", "therefore", "as a result",
                           "given that", "considering", "kinematic", "collision",
                           "closing", "speed", "distance", "trajectory"]
        causal_count = sum(1 for kw in causal_keywords if kw in prediction_lower)

        if causal_count >= 4:
            reasoning_score = 2.0
        elif causal_count >= 2:
            reasoning_score = 1.5
        elif causal_count >= 1:
            reasoning_score = 1.0
        elif len(prediction_output.strip()) > 50:
            reasoning_score = 0.5  # At least some text was produced

        node.metrics.append(MetricScore(
            name="Reasoning Quality",
            score=round(min(reasoning_score, 2.0), 1),
            max_score=2.0,
            detail=f"{causal_count} causal reasoning indicators found",
        ))

        return node

    @staticmethod
    def _extract_prediction_objects(text: str) -> list[dict]:
        """Try to extract structured prediction objects from text."""
        # Try JSON markdown block
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "scene_prediction" in data:
                    return data["scene_prediction"]
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict) and "scene_prediction" in data:
                    return data["scene_prediction"]
            except json.JSONDecodeError:
                pass

        return []

    @staticmethod
    def _risk_matches(actual: str, expected: str) -> bool:
        """Check if actual risk level matches expected."""
        actual = actual.upper().strip()
        if "-" in expected:
            # Range like "MODERATE-HIGH"
            options = [o.strip() for o in expected.split("-")]
            return actual in options
        return actual == expected

    @staticmethod
    def _risk_close(actual: str, expected: str) -> bool:
        """Check if risk is close (one level off)."""
        levels = ["NO RISK", "NONE", "LOW", "MODERATE", "HIGH"]
        actual = actual.upper().strip()

        actual_idx = -1
        expected_idx = -1
        for i, lvl in enumerate(levels):
            if lvl in actual:
                actual_idx = i
            if lvl in expected:
                expected_idx = i

        if actual_idx >= 0 and expected_idx >= 0:
            return abs(actual_idx - expected_idx) <= 1
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Planning Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class PlanningEvaluator:
    """
    Scores planning output against a physics-based gold standard action.

    Metrics (10 points total):
      Action Safety         (4) — Is the action safe?
      Action Appropriateness(3) — Is it proportional?
      Reasoning Coherence   (3) — Does reasoning support the action?
    """

    # Gold standard action derivation
    @staticmethod
    def _derive_gold_action(
        annotations: dict,
        perception_result: PerceptionResult,
    ) -> str:
        """Derive the physics-based gold standard action."""
        ego_speed = annotations.get("ego_speed", 0.0)

        # Get front objects
        front_objects = []
        for vp in perception_result.views.values():
            if vp.view_name == "Front":
                front_objects.extend(vp.detections)

        if not front_objects:
            if ego_speed < 0.01:
                return "ACCELERATE"
            return "KEEP_SPEED"

        # Sort by distance
        front_sorted = sorted(front_objects, key=lambda d: d.get("dist", float("inf")))
        closest = front_sorted[0]
        closest_dist = closest.get("dist", 999)
        closest_speed = closest.get("speed", 0.0)

        if ego_speed < 0.01:
            # Already stopped
            if closest_dist < 15.0:
                return "KEEP_SPEED"  # Stay stopped
            return "ACCELERATE"

        if closest_dist < 10.0 and closest_speed < 0.5:
            return "EMERGENCY_BRAKE"
        elif closest_dist < 15.0:
            return "STOP"
        elif closest_dist < 30.0:
            return "DECELERATE"
        else:
            return "KEEP_SPEED"

    @staticmethod
    def evaluate(
        planning_result: dict,
        prediction_output: str,
        perception_result: PerceptionResult,
        annotations: dict,
    ) -> NodeScore:
        """Score the planning output."""
        node = NodeScore(node_name="Planning (Pl)")

        action = planning_result.get("selected_action", "UNKNOWN").upper()
        reasoning = planning_result.get("planning_reasoning", "")
        reasoning_lower = reasoning.lower()
        ego_speed = annotations.get("ego_speed", 0.0)

        gold_action = PlanningEvaluator._derive_gold_action(annotations, perception_result)

        # Define action severity ranking (higher = more cautious)
        action_severity = {
            "ACCELERATE": 0,
            "KEEP_SPEED": 1,
            "DECELERATE": 2,
            "STOP": 3,
            "EMERGENCY_BRAKE": 4,
            "STEER_TO_AVOID": 3,
        }
        gold_severity = action_severity.get(gold_action, 2)
        actual_severity = action_severity.get(action, 2)

        # For VLM_DECIDED (baseline), try to extract action from reasoning
        if action == "VLM_DECIDED":
            extracted = PlanningEvaluator._extract_action_from_text(reasoning)
            if extracted:
                action = extracted
                actual_severity = action_severity.get(action, 2)

        # ── Metric 1: Action Safety (4 points) ──────────────────────────
        # Safe = not dangerously under-reacting
        safety_score = 0.0

        if gold_severity >= 3:  # Gold says STOP or EMERGENCY_BRAKE
            if actual_severity >= 3:
                safety_score = 4.0  # Correctly cautious
            elif actual_severity >= 2:
                safety_score = 2.0  # Somewhat cautious
            elif actual_severity >= 1:
                safety_score = 1.0  # Insufficiently cautious
            else:
                safety_score = 0.0  # Dangerously under-reacting
        elif gold_severity >= 2:  # Gold says DECELERATE
            if actual_severity >= 2:
                safety_score = 4.0
            elif actual_severity >= 1:
                safety_score = 3.0
            else:
                safety_score = 1.0
        elif gold_severity <= 1:  # Gold says KEEP_SPEED or ACCELERATE
            if actual_severity <= 2:
                safety_score = 4.0  # Not over-reacting
            elif actual_severity == 3:
                safety_score = 3.0  # Slightly over-cautious (still safe)
            else:
                safety_score = 2.0  # Over-reacting (unnecessary emergency brake)

        node.metrics.append(MetricScore(
            name="Action Safety",
            score=round(min(safety_score, 4.0), 1),
            max_score=4.0,
            detail=f"Action={action}, Gold={gold_action}",
        ))

        # ── Metric 2: Action Appropriateness (3 points) ──────────────────
        # Proportional = matches gold action closely
        severity_diff = abs(gold_severity - actual_severity)

        if severity_diff == 0:
            appropriateness_score = 3.0  # Perfect match
        elif severity_diff == 1:
            appropriateness_score = 2.0  # One level off
        elif severity_diff == 2:
            appropriateness_score = 1.0  # Two levels off
        else:
            appropriateness_score = 0.0  # Way off

        node.metrics.append(MetricScore(
            name="Action Appropriateness",
            score=round(min(appropriateness_score, 3.0), 1),
            max_score=3.0,
            detail=f"Severity diff: {severity_diff} (actual={actual_severity}, gold={gold_severity})",
        ))

        # ── Metric 3: Reasoning Coherence (3 points) ─────────────────────
        coherence_score = 0.0

        # Check if reasoning mentions relevant scene elements
        mentions_action = any(kw in reasoning_lower for kw in [
            "brake", "stop", "slow", "decelerate", "maintain", "keep speed",
            "accelerate", "speed up", "emergency", "steer", "avoid"
        ])
        mentions_reason = any(kw in reasoning_lower for kw in [
            "object", "vehicle", "car", "pedestrian", "walker",
            "distance", "ahead", "front", "close", "far",
            "speed", "stopped", "moving", "approaching"
        ])
        mentions_logic = any(kw in reasoning_lower for kw in [
            "because", "since", "therefore", "due to", "given",
            "risk", "safe", "danger", "threat", "collision"
        ])

        if mentions_action:
            coherence_score += 1.0
        if mentions_reason:
            coherence_score += 1.0
        if mentions_logic:
            coherence_score += 1.0

        # If reasoning is very short or empty, cap low
        if len(reasoning.strip()) < 10:
            coherence_score = 0.0

        node.metrics.append(MetricScore(
            name="Reasoning Coherence",
            score=round(min(coherence_score, 3.0), 1),
            max_score=3.0,
            detail=f"Action={mentions_action}, Context={mentions_reason}, Logic={mentions_logic}",
        ))

        return node

    @staticmethod
    def _extract_action_from_text(text: str) -> Optional[str]:
        """Try to extract a driving action from free-form text."""
        text_lower = text.lower()

        # Check in order of severity (highest first)
        if any(kw in text_lower for kw in ["emergency brake", "emergency stop", "emergency braking"]):
            return "EMERGENCY_BRAKE"
        if any(kw in text_lower for kw in ["stop immediately", "come to a stop", "full stop"]):
            return "STOP"
        if any(kw in text_lower for kw in ["steer", "swerve", "lane change", "avoid"]):
            return "STEER_TO_AVOID"
        if any(kw in text_lower for kw in ["slow down", "decelerate", "reduce speed", "brake"]):
            return "DECELERATE"
        if any(kw in text_lower for kw in ["maintain speed", "keep speed", "continue", "keep driving"]):
            return "KEEP_SPEED"
        if any(kw in text_lower for kw in ["accelerate", "speed up"]):
            return "ACCELERATE"

        return None


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Runner — Orchestrates full comparison
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationRunner:
    """
    Runs the full baseline vs optimized evaluation across one or more frames.
    """

    @staticmethod
    def evaluate_pipeline(
        pipeline_name: str,
        perception_result: PerceptionResult,
        prediction_output: str,
        planning_result: dict,
        annotations: dict,
        radar_anchors: list[dict],
    ) -> PipelineScore:
        """Score a single pipeline run."""
        score = PipelineScore(pipeline_name=pipeline_name)

        score.perception = PerceptionEvaluator.evaluate(
            perception_result, annotations, radar_anchors
        )
        score.prediction = PredictionEvaluator.evaluate(
            prediction_output, perception_result, annotations
        )
        score.planning = PlanningEvaluator.evaluate(
            planning_result, prediction_output, perception_result, annotations
        )

        return score

    @staticmethod
    def print_comparison(
        frame_id: str,
        baseline_score: PipelineScore,
        optimized_score: PipelineScore,
    ) -> str:
        """Print a formatted comparison table and return it as a string."""
        W = 66

        lines = []
        lines.append(f"╔{'═' * W}╗")
        lines.append(f"║  EVALUATION REPORT — Frame {frame_id:<{W - 30}}║")
        lines.append(f"╠{'═' * W}╣")

        header = f"║  {'Metric':<28}{'BASELINE':>12}{'OPTIMIZED':>12}{'':>{W - 52 - 2}}║"
        lines.append(header)
        lines.append(f"╠{'═' * W}╣")

        # Print each node
        for b_node, o_node in [
            (baseline_score.perception, optimized_score.perception),
            (baseline_score.prediction, optimized_score.prediction),
            (baseline_score.planning, optimized_score.planning),
        ]:
            lines.append(f"║  \033[1m{b_node.node_name:<{W - 4}}\033[0m║")

            for b_m, o_m in zip(b_node.metrics, o_node.metrics):
                b_str = f"{b_m.score:.1f}/{b_m.max_score:.0f}"
                o_str = f"{o_m.score:.1f}/{o_m.max_score:.0f}"
                line = f"║    {b_m.name:<26}{b_str:>12}{o_str:>12}{'':>{W - 54}}║"
                lines.append(line)

            b_sub = f"{b_node.total:.1f}/{b_node.max_total:.0f}"
            o_sub = f"{o_node.total:.1f}/{o_node.max_total:.0f}"
            lines.append(f"║    {'SUBTOTAL':<26}{b_sub:>12}{o_sub:>12}{'':>{W - 54}}║")
            lines.append(f"╠{'═' * W}╣")

        # Total
        b_total = f"{baseline_score.total:.1f}/{baseline_score.max_total:.0f}"
        o_total = f"{optimized_score.total:.1f}/{optimized_score.max_total:.0f}"
        lines.append(f"║  \033[1m{'TOTAL':<26}\033[0m{b_total:>14}{o_total:>12}{'':>{W - 54}}║")
        lines.append(f"╚{'═' * W}╝")

        # Detail breakdown
        lines.append("")
        lines.append("  DETAIL BREAKDOWN:")
        lines.append(f"  {'─' * 60}")

        for label, score in [("Baseline", baseline_score), ("Optimized", optimized_score)]:
            lines.append(f"\n  [{label}]")
            for node in [score.perception, score.prediction, score.planning]:
                for m in node.metrics:
                    indicator = "✓" if m.score >= m.max_score * 0.7 else ("~" if m.score >= m.max_score * 0.4 else "✗")
                    lines.append(f"    {indicator} {m.name}: {m.score:.1f}/{m.max_score:.0f} — {m.detail}")

        output = "\n".join(lines)
        print(output)
        return output
