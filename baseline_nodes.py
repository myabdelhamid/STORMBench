"""
StormVLM — Baseline (Pre-Optimization) Nodes
===============================================
Master's Thesis · GIU Berlin
Author: Marwan Elsayed

These are the BASELINE versions of the 3 pipeline nodes, using basic/naive
prompts and returning raw text (not JSON).  They exist for before/after
prompt-optimization evaluation.

Differences from the optimized nodes:
  • Simple, unstructured prompts (no JSON schema, no engineered instructions)
  • Raw text output (no JSON parsing or validation)
  • No post-processing rules (no close-range overrides, no physics fixes)
  • No CLAHE fog enhancement on images
  • Radar data is listed but not pre-matched to objects
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np  # type: ignore[import]
from PIL import Image  # type: ignore[import]

from perception_node import (
    AnnotationLoader,
    PerceptionResult,
    ViewPerception,
    _MAX_IMAGE_SIZE,
    _VIEW_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="[StormVLM %(levelname)s] %(message)s",
)
log = logging.getLogger("stormvlm.baseline")


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Perception Node
# ═══════════════════════════════════════════════════════════════════════════

class BaselinePerceptionNode:
    """
    Baseline perception node with a simple, unoptimized prompt.
    Returns raw text — no JSON, no categorization, no post-processing.
    """

    _CAMERA_TO_VIEW: dict[str, str] = {
        "Camera 0 (Front)": "Front",
        "Camera 1 (Right)": "Right",
        "Camera 2 (Left)": "Left",
        "Camera 3 (Back)": "Back",
    }

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
        max_words: int = 150,
        temperature: float = 0.1,
    ) -> None:
        self.model_id = model_id
        self.max_words = max_words
        self.temperature = temperature
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        log.info(f"[BaselinePerception] Loading {self.model_id} ...")
        from mlx_vlm import load  # type: ignore[import]
        self._model, self._processor = load(self.model_id)
        log.info("[BaselinePerception] Model loaded ✓")

    def perceive_frame(
        self,
        data_dir: str | Path,
        frame_id: str,
        images: Optional[list[Image.Image]] = None,
        radar_anchors: Optional[list[dict]] = None,
    ) -> PerceptionResult:
        """Run baseline perception — simple prompt, raw text output."""
        self._ensure_model()
        data_dir = Path(data_dir)

        # 1. Load YAML annotations (same input data as optimized)
        yaml_path = data_dir / f"{frame_id}.yaml"
        annotations = AnnotationLoader.load(yaml_path)

        # 2. Load camera images
        if images is None:
            images = self._load_cameras(data_dir, frame_id)

        # 3. Load radar anchors
        if radar_anchors is None:
            from stormvlm_loader import GlobalRadarFilter
            rf = GlobalRadarFilter()
            radar_anchors = rf.process(str(data_dir), frame_id)

        # 4. Group by view
        view_map = self._group_by_view(annotations, radar_anchors)

        # 5. Get weather
        weather_val = annotations.get("weather_type", "fd").lower()
        if "cd" in weather_val:
            scenario_str = "clear day"
        elif "fd" in weather_val:
            scenario_str = "heavy fog"
        elif "fhrd" in weather_val:
            scenario_str = "heavy fog and rain"
        else:
            scenario_str = "heavy fog"

        result = PerceptionResult(
            frame_id=frame_id,
            ego_speed=annotations.get("ego_speed", 0.0),
            weather_type=weather_val,
        )

        view_to_cam = {"Front": 0, "Right": 1, "Left": 2, "Back": 3}

        for view_name in _VIEW_NAMES:
            cam_idx = view_to_cam[view_name]
            detections = view_map.get(view_name, {}).get("detections", [])
            anchors = view_map.get(view_name, {}).get("anchors", [])

            # Build BASIC prompt (no JSON schema, no engineered instructions)
            prompt = self._build_prompt(
                view_name, detections, anchors, scenario=scenario_str
            )
            log.info(
                f"[BaselinePerception] {view_name} view: "
                f"{len(detections)} detections, {len(anchors)} anchors "
                f"→ querying LVLM..."
            )

            # Save temp image (NO CLAHE enhancement — just resize)
            temp_path = self._save_temp_image(images[cam_idx])
            reasoning = self._query_model(temp_path, prompt)

            try:
                os.unlink(temp_path)
            except OSError:
                pass

            # NO post-processing — raw text output as-is
            result.views[view_name] = ViewPerception(
                view_name=view_name,
                detections=detections,
                radar_anchors=anchors,
                reasoning=reasoning,
            )

        return result

    @staticmethod
    def _load_cameras(data_dir: Path, frame_id: str) -> list[Image.Image]:
        images = []
        for i in range(4):
            matches = glob.glob(str(data_dir / f"{frame_id}_camera{i}.*"))
            if matches:
                images.append(Image.open(matches[0]).convert("RGB"))
            else:
                log.warning(f"Camera {i} not found for frame {frame_id}")
                images.append(Image.new("RGB", (800, 600), (128, 128, 128)))
        return images

    @staticmethod
    def _save_temp_image(img: Image.Image) -> str:
        """Resize and save — NO CLAHE enhancement (baseline)."""
        resized = img.resize(_MAX_IMAGE_SIZE, Image.LANCZOS)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        resized.save(path)
        return path

    @staticmethod
    def _group_by_view(
        annotations: dict, radar_anchors: list[dict]
    ) -> dict[str, dict]:
        grouped: dict[str, dict] = {
            v: {"detections": [], "anchors": []}
            for v in _VIEW_NAMES
        }
        for obj in annotations.get("vehicles", []) + annotations.get("walkers", []):
            view = obj.get("view", "Front")
            if view in grouped:
                grouped[view]["detections"].append(obj)
        for anchor in radar_anchors:
            primary = BaselinePerceptionNode._CAMERA_TO_VIEW.get(
                anchor.get("primary_camera", ""), None
            )
            if primary and primary in grouped:
                grouped[primary]["anchors"].append(anchor)
        return grouped

    @staticmethod
    def _build_prompt(
        view_name: str,
        detections: list[dict[str, Any]],
        anchors: list[dict[str, Any]],
        scenario: str = "heavy fog",
    ) -> str:
        """
        Build a BASIC, unoptimized prompt.
        No JSON schema, no categorization instructions, no engineered structure.
        """
        prompt = f"You are looking at the {view_name.lower()} camera of a vehicle driving in {scenario}.\n\n"

        # List detected objects simply
        if detections:
            prompt += "The following objects were detected by sensors:\n"
            for d in detections:
                cls_name = d.get("class", "object")
                dist = d.get("dist", 0.0)
                speed = d.get("speed", 0.0)
                prompt += f"- A {cls_name} at {dist:.1f} meters, speed {speed:.1f} m/s\n"
        else:
            prompt += "No objects were detected by sensors.\n"

        # List radar data simply
        if anchors:
            prompt += "\nRadar data:\n"
            for a in anchors:
                prompt += f"- Radar contact at {a.get('dist_m', 0.0):.1f}m, speed {a.get('max_speed_ms', 0.0):.1f} m/s\n"
        else:
            prompt += "\nNo radar data available.\n"

        prompt += "\nDescribe what you see in the image in plain English. For each detected object, say whether it is visible in the camera and where it is located in the scene (on the road, sidewalk, etc).\n"
        prompt += "IMPORTANT: Write your answer as plain English sentences only. Do NOT use JSON, code blocks, or any structured format.\n"
        prompt += "Answer:"

        return prompt

    def _query_model(self, image_path: str, prompt: str) -> str:
        from mlx_vlm import generate  # type: ignore[import]
        result = generate(
            self._model,
            self._processor,
            prompt=prompt,
            image=[image_path],
            max_tokens=300,
            temperature=self.temperature,
            verbose=False,
            resize_shape=_MAX_IMAGE_SIZE,
        )
        raw = (
            result.text.strip()
            if hasattr(result, "text")
            else str(result).strip()
        )
        return self._clean_raw_output(raw, self.max_words)

    @staticmethod
    def _clean_raw_output(text: str, max_words: int = 150) -> str:
        """
        Clean VLM output: strip JSON/code artifacts, deduplicate repeated
        blocks, and enforce word budget.  The goal is plain English text.
        """
        # 1. Strip markdown code fences and JSON blocks entirely
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # 2. If the VLM still returned JSON, extract readable parts
        if text.strip().startswith('{'):
            try:
                data = json.loads(text.strip())
                # Convert JSON to plain English
                parts = []
                if isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, list):
                            for item in val:
                                if isinstance(item, dict):
                                    obj_type = item.get('type', item.get('class', 'object'))
                                    dist = item.get('distance', item.get('dist', '?'))
                                    visible = item.get('visible', '?')
                                    parts.append(f"A {obj_type} at {dist}m, visible: {visible}.")
                        elif isinstance(val, str):
                            parts.append(val)
                if parts:
                    text = ' '.join(parts)
            except (json.JSONDecodeError, TypeError):
                pass

        # 3. Deduplicate: split into sentences and remove repeats
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen: set[str] = set()
        unique: list[str] = []
        for sent in sentences:
            key = sent.strip().lower()
            if not key or len(key) < 5:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append(sent.strip())

        text = ' '.join(unique)

        # 4. Remove any leftover JSON-like fragments
        text = re.sub(r'[{}\[\]]', '', text)
        text = re.sub(r'"\w+"\s*:', '', text)  # key: patterns
        text = re.sub(r'\s+', ' ', text).strip()

        # 5. Enforce word budget
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
            if text and text[-1] not in '.!?':
                text += '.'

        return text if text else "No observations available."


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Prediction Node
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BaselinePredictionResult:
    """Prediction result — raw text."""
    frame_id: str
    prediction_text: str = ""

    def full_report(self) -> str:
        lines = [
            f"{'='*60}",
            f"BASELINE PREDICTION REPORT — Frame {self.frame_id}",
            f"{'='*60}",
            self.prediction_text,
            f"\n{'='*60}",
        ]
        return "\n".join(lines)


class BaselinePredictionNode:
    """
    Baseline prediction node with a simple prompt.
    Takes perception text (not parsed JSON) and asks for plain-text predictions.
    No physics-based post-processing.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
        temperature: float = 0.1,
        model=None,
        processor=None,
        max_words: int = 150,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self._model = model
        self._processor = processor
        self.max_words = max_words

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        log.info(f"[BaselinePrediction] Loading {self.model_id} ...")
        from mlx_vlm import load  # type: ignore[import]
        self._model, self._processor = load(self.model_id)
        log.info("[BaselinePrediction] Model loaded ✓")

    def predict_frame(
        self,
        perception_result: PerceptionResult,
        data_dir: str | Path,
        images: Optional[list[Image.Image]] = None,
    ) -> BaselinePredictionResult:
        """Run baseline prediction — simple prompt, raw text output."""
        self._ensure_model()
        data_dir = Path(data_dir)

        # Load front camera image
        if images is None:
            matches = glob.glob(str(data_dir / f"{perception_result.frame_id}_camera0.*"))
            if matches:
                img = Image.open(matches[0]).convert("RGB")
            else:
                img = Image.new("RGB", (800, 600), (128, 128, 128))
        else:
            img = images[0]

        # Build a CONCISE perception summary (not raw VLM text — that's too noisy)
        perception_summary = ""
        for view_name, vp in perception_result.views.items():
            if not vp.detections:
                continue
            perception_summary += f"\n{view_name} camera ({len(vp.detections)} objects):\n"
            for d in vp.detections:
                cls_name = d.get("class", "object")
                dist = d.get("dist", 0.0)
                speed = d.get("speed", 0.0)
                status = "stopped" if speed < 0.5 else f"moving at {speed:.1f} m/s"
                perception_summary += f"  - {cls_name} at {dist:.1f}m, {status}\n"
            # Add a brief summary of VLM observation (first 30 words max)
            obs_words = vp.reasoning.split()[:30]
            if obs_words:
                brief_obs = ' '.join(obs_words)
                if brief_obs and brief_obs[-1] not in '.!?':
                    brief_obs += '...'
                perception_summary += f"  Observation: {brief_obs}\n"

        prompt = self._build_prompt(perception_summary, perception_result.ego_speed)

        temp_path = self._save_temp_image(img)
        log.info("[BaselinePrediction] Querying LVLM...")
        reasoning = self._query_model(temp_path, prompt)

        try:
            os.unlink(temp_path)
        except OSError:
            pass

        # NO post-processing — return raw text
        return BaselinePredictionResult(
            frame_id=perception_result.frame_id,
            prediction_text=reasoning,
        )

    def _save_temp_image(self, img: Image.Image) -> str:
        resized = img.resize(_MAX_IMAGE_SIZE, Image.LANCZOS)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        resized.save(path)
        return path

    @staticmethod
    def _build_prompt(perception_summary: str, ego_speed: float) -> str:
        """Basic, unoptimized prediction prompt."""
        prompt = (
            f"You are analyzing a driving scene. The ego vehicle is moving at {ego_speed:.1f} m/s.\n\n"
            f"Here is what was detected around the vehicle:\n"
            f"{perception_summary}\n\n"
            f"For each object, predict what it will do next and how dangerous it is for the ego vehicle.\n"
            f"IMPORTANT: Write your answer as plain English sentences only. Do NOT use JSON, code blocks, or any structured format.\n"
            f"Answer:"
        )
        return prompt

    def _query_model(self, image_path: str, prompt: str) -> str:
        from mlx_vlm import generate  # type: ignore[import]
        result = generate(
            self._model,
            self._processor,
            prompt=prompt,
            image=[image_path],
            max_tokens=400,
            temperature=self.temperature,
            verbose=False,
            resize_shape=_MAX_IMAGE_SIZE,
        )
        raw = result.text.strip() if hasattr(result, "text") else str(result).strip()
        return BaselinePerceptionNode._clean_raw_output(raw, self.max_words)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Planning Node
# ═══════════════════════════════════════════════════════════════════════════

class BaselinePlanningNode:
    """
    Baseline planning node with a simple prompt.
    No deterministic action selection — everything is left to the VLM.
    Returns raw text.
    """

    def __init__(
        self,
        model=None,
        processor=None,
        model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
        max_words: int = 120,
        temperature: float = 0.1,
    ) -> None:
        self.model_id = model_id
        self.max_words = max_words
        self.temperature = temperature
        self._model = model
        self._processor = processor

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        log.info(f"[BaselinePlanning] Loading {self.model_id} ...")
        from mlx_vlm import load  # type: ignore[import]
        self._model, self._processor = load(self.model_id)
        log.info("[BaselinePlanning] Model loaded ✓")

    def plan_action(
        self, prediction_text: str, ego_speed: float
    ) -> dict:
        """Run baseline planning — simple prompt, raw text output."""
        self._ensure_model()

        # Truncate prediction text if too long (prevent context overflow)
        pred_words = prediction_text.split()
        if len(pred_words) > 200:
            prediction_text = ' '.join(pred_words[:200]) + '...'

        prompt = (
            f"You are driving a car at {ego_speed:.1f} m/s.\n\n"
            f"Here is what is happening around you:\n"
            f"{prediction_text}\n\n"
            f"What should the driver do? State the recommended action (e.g., brake, slow down, keep speed, accelerate) and explain your reasoning in one or two sentences.\n"
            f"IMPORTANT: Write your answer as plain English sentences only. Do NOT use JSON, code blocks, or any structured format.\n"
            f"Answer:"
        )

        try:
            from mlx_vlm import generate  # type: ignore[import]

            # Use a tiny placeholder image (mlx_vlm requires one)
            img = Image.new("RGB", (64, 64), color="black")
            fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            img.save(temp_path)

            result = generate(
                self._model,
                self._processor,
                prompt=prompt,
                image=[temp_path],
                max_tokens=150,
                temperature=self.temperature,
                verbose=False,
            )
            raw = result.text.strip() if hasattr(result, "text") else str(result).strip()

            try:
                os.unlink(temp_path)
            except OSError:
                pass

            # Clean up output
            raw = BaselinePerceptionNode._clean_raw_output(raw, self.max_words)

            return {
                "selected_action": "VLM_DECIDED",
                "planning_reasoning": raw,
            }

        except Exception as e:
            log.warning(f"[BaselinePlanning] VLM failed ({e}); using fallback.")
            return {
                "selected_action": "UNKNOWN",
                "planning_reasoning": f"VLM inference failed: {e}",
            }
