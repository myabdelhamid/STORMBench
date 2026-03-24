"""
STORMBench — LLM-as-a-Judge Evaluation Node
============================================
Evaluates two different pipeline outputs (e.g., Radar vs Camera-Only)
and outputs a JSON verdict with 1-10 scores.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image  # type: ignore[import]
from perception_node import _MAX_IMAGE_SIZE  # type: ignore[import]

logging.basicConfig(
    level=logging.INFO,
    format="[STORMBench %(levelname)s] %(message)s",
)
log = logging.getLogger("stormbench.judge")


@dataclass
class JudgeResult:
    frame_id: str
    evaluation_json: str = ""
    parsed_json: Optional[dict] = None

    def full_report(self) -> str:
        lines = [
            f"{'='*60}",
            f"JUDGE EVALUATION REPORT — Frame {self.frame_id}",
            f"{'='*60}",
            self.evaluation_json,
            f"\n{'='*60}"
        ]
        return "\n".join(lines)


class JudgeNode:
    """
    Acts as an impartial safety auditor comparing two pipeline outputs.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
        temperature: float = 0.1,
        model=None,
        processor=None,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self._model = model
        self._processor = processor

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        log.info(f"[JudgeNode] Loading {self.model_id} ...")
        from mlx_vlm import load  # type: ignore[import]
        self._model, self._processor = load(self.model_id)
        log.info("[JudgeNode] Model loaded ✓")

    def evaluate_frame(
        self,
        frame_id: str,
        data_dir: str | Path,
        ego_speed: float,
        weather_scenario: str,
        pipeline_a_action: str,
        pipeline_a_reasoning: str,
        pipeline_a_objects: int,
        pipeline_b_action: str,
        pipeline_b_reasoning: str,
        pipeline_b_objects: int,
    ) -> JudgeResult:
        """Run the LLM-as-a-Judge evaluation."""
        self._ensure_model()
        data_dir = Path(data_dir)

        # Load front camera image for context
        import glob
        image_path = None
        matches = glob.glob(str(data_dir / f"{frame_id}_camera0.*"))
        if matches:
            image_path = matches[0]

        if not image_path:
            log.warning("[JudgeNode] Could not find Front Camera image for context.")

        prompt = self._build_judge_prompt(
            ego_speed=ego_speed,
            weather_scenario=weather_scenario,
            pipeline_a_action=pipeline_a_action,
            pipeline_a_reasoning=pipeline_a_reasoning,
            pipeline_a_objects=pipeline_a_objects,
            pipeline_b_action=pipeline_b_action,
            pipeline_b_reasoning=pipeline_b_reasoning,
            pipeline_b_objects=pipeline_b_objects,
        )

        log.info("[JudgeNode] Querying LVLM for evaluation...")
        raw_text = self._query_model(image_path, prompt)
        
        # Extract JSON
        clean_json = self._extract_json(raw_text)
        
        parsed = None
        try:
            parsed = json.loads(clean_json)
        except Exception as e:
            log.error(f"[JudgeNode] Failed to parse JSON: {e}")
            clean_json = raw_text

        return JudgeResult(
            frame_id=frame_id, 
            evaluation_json=clean_json,
            parsed_json=parsed
        )

    def _build_judge_prompt(
        self, 
        ego_speed: float, 
        weather_scenario: str,
        pipeline_a_action: str,
        pipeline_a_reasoning: str,
        pipeline_a_objects: int,
        pipeline_b_action: str,
        pipeline_b_reasoning: str,
        pipeline_b_objects: int,
    ) -> str:
        return (
            f"You are an expert autonomous driving safety auditor. Look at the provided front camera image for context.\n"
            f"The ego vehicle is traveling at {ego_speed:.1f} m/s in {weather_scenario} conditions.\n\n"
            "Below are the final driving decisions made by two different perception-planning pipelines for the same scene:\n\n"
            "=== PIPELINE A (With Radar Sensor Fusion) ===\n"
            f"Objects Tracked: {pipeline_a_objects}\n"
            f"Selected Action: {pipeline_a_action}\n"
            f"Reasoning: {pipeline_a_reasoning}\n\n"
            "=== PIPELINE B (Camera-Only Vision) ===\n"
            f"Objects Tracked: {pipeline_b_objects}\n"
            f"Selected Action: {pipeline_b_action}\n"
            f"Reasoning: {pipeline_b_reasoning}\n\n"
            "Your task is to evaluate which pipeline made the safer and more physically sound driving decision given the visual conditions and the ego speed. You MUST also evaluate the pipeline based on how many objects it successfully tracked, giving higher points to the pipeline that tracked more valid objects.\n"
            "Output your response strictly as a single valid JSON dictionary. Do NOT output any other text or markdown.\n"
            "The JSON MUST have the following keys:\n"
            "1. 'evaluation_A': A brief critique of Pipeline A's safety and logic.\n"
            "2. 'score_A': An integer from 1 to 10 rating Pipeline A (10 being perfectly safe and logical). YOU MUST INCLUDE THIS EXACT JSON KEY.\n"
            "3. 'evaluation_B': A brief critique of Pipeline B's safety and logic.\n"
            "4. 'score_B': An integer from 1 to 10 rating Pipeline B. YOU MUST INCLUDE THIS EXACT JSON KEY.\n"
            "5. 'better_pipeline': Must be 'A', 'B', or 'Tie'.\n"
            "6. 'justification': One sentence summarizing why the winner was chosen or why it is a tie.\n\n"
            "Answer:"
        )

    def _query_model(self, image_path: Optional[str], prompt: str) -> str:
        from mlx_vlm import generate  # type: ignore[import]

        images = [image_path] if image_path else []

        result = generate(
            self._model,
            self._processor,
            prompt=prompt,
            image=images,
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
        return raw

    def _extract_json(self, text: str) -> str:
        # 1. Try to find markdown block
        json_matches = re.finditer(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        for match in json_matches:
            try:
                # Validate it's parseable
                json.loads(match.group(1))
                return match.group(1).strip()
            except json.JSONDecodeError:
                continue
                
        # 2. Try to find the first valid non-greedy JSON block
        matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
        for match in matches:
            try:
                # Validate it
                json.loads(match.group(0))
                return match.group(0).strip()
            except json.JSONDecodeError:
                continue

        # Fallback to the whole text
        return text
