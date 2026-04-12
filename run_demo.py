"""
StormVLM — Phase I Demo & Smoke Test
=======================================
Generates a synthetic dataset layout (default) OR loads real flat-file data
(--real), runs both ablation conditions, prints the reasoning logs, and
optionally runs MiniCPM-V-2_6 via mlx-vlm for a real model response.

Usage
-----
  python run_demo.py                          # synthetic smoke-test (no model)
  python run_demo.py --baseline               # camera-only baseline (synthetic)
  python run_demo.py --frame 0002             # use a specific frame ID (synthetic)

  python run_demo.py --real                   # real data, frame auto-detected
  python run_demo.py --real --frame 000436    # real data, explicit frame ID
  python run_demo.py --real --baseline        # real data, camera-only ablation

  # DriveLM-style perception (YAML annotations + radar + VLM reasoning)
  python run_demo.py --real --perceive                    # auto-detect frame
  python run_demo.py --real --perceive --frame 000514     # specific frame

Real data layout expected
--------------------------
  <data-root>/
    {frame_id}_camera0.png   (front)
    {frame_id}_camera1.png   (back)
    {frame_id}_camera2.png   (left)
    {frame_id}_camera3.png   (right)
    {frame_id}_radar0.npy
    ...
    {frame_id}_radar5.npy

Model
-----
  mlx-community/Qwen2.5-VL-3B-Instruct-4bit  (runs locally on Apple Silicon via mlx-vlm)
  Qwen2-VL-2B is replaced by Qwen2.5-VL-3B which has correct 4-bit quantisation support
  in mlx-vlm 0.3.x. The model is loaded once and reused for all ablation conditions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np # type: ignore[import]
from PIL import Image

from stormvlm_loader import FrameData, StormVLMLoader # type: ignore[import]


# =============================================================================
# Real flat-file loader (bypasses StormVLMLoader's nested-dir expectation)
# =============================================================================

CAMERA_ORDER = ["front", "back", "left", "right"]  # camera0 → camera3


# Qwen2-VL patch size is 14px; merged patches need dims divisible by 28.
# 448 = 28 × 16 — gives 4× more visual detail than 224×224.
# Each 448×448 image → ~256 merged patches → ~256 tokens; 4 cameras ≈ 1024 tokens total,
# well within the 32768 context limit.
_MAX_IMAGE_SIZE = (448, 448)


def _resize_for_model(img: Image.Image, size: tuple[int, int] = _MAX_IMAGE_SIZE) -> Image.Image:
    """Resize (and crop-pad) image to exactly `size` for Qwen2-VL-4bit compatibility."""
    return img.resize(size, Image.LANCZOS)


def _load_real_frame(data_root: Path, frame_id: str, use_radar: bool = True) -> FrameData:
    """
    Load real data from flat files:
      {frame_id}_camera{0-3}.png  and  {frame_id}_radar{0-5}.npy

    Returns a FrameData object identical in structure to what StormVLMLoader
    would return, so the rest of the pipeline (RadarFusion, prompt builder)
    works unchanged.
    """
    frame = FrameData(frame_id=frame_id)

    # ── Cameras ───────────────────────────────────────────────────────────
    for i in range(4):
        # try both .png and .jpg
        found = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = data_root / f"{frame_id}_camera{i}{ext}"
            if p.exists():
                found = p
                break
        if found is None:
            print(f"[WARN] Camera {i} not found, using black placeholder")
            frame.images.append(Image.new("RGB", (224, 224)))
        else:
            frame.images.append(_resize_for_model(Image.open(found).convert("RGB")))

    # ── Radars ────────────────────────────────────────────────────────────
    if use_radar:
        for i in range(6):
            p = data_root / f"{frame_id}_radar{i}.npy"
            if not p.exists():
                print(f"[WARN] Radar {i} not found, using empty array")
                frame.radar_arrays.append(np.empty((0, 4), dtype=np.float32))
            else:
                arr = np.load(str(p), mmap_mode="r")
                if arr.ndim != 2 or arr.shape[1] != 4:
                    print(f"[WARN] Unexpected radar shape {arr.shape} in {p}")
                    frame.radar_arrays.append(np.empty((0, 4), dtype=np.float32))
                else:
                    frame.radar_arrays.append(arr)

    return frame


def _detect_frame_id(data_root: Path) -> str:
    """Auto-detect the frame ID from the first camera file found."""
    search_dirs = [data_root, data_root / "dataset"]
    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.name.endswith("_camera0.png") or f.name.endswith("_camera0.jpg"):
                return f.name.split("_camera0")[0]
    raise FileNotFoundError(
        f"No *_camera0.png files found in {data_root}. "
        "Pass --frame <frame_id> explicitly."
    )


# =============================================================================
# Synthetic dataset factory (unchanged — used when --real is NOT passed)
# =============================================================================

def _make_synthetic_dataset(root: Path, frame_id: str = "0001") -> Path:
    frame_dir = root / frame_id
    cam_dir = frame_dir / "cameras"
    rad_dir = frame_dir / "radars"
    cam_dir.mkdir(parents=True, exist_ok=True)
    rad_dir.mkdir(parents=True, exist_ok=True)

    colours = {
        "front": (200, 50, 50),
        "back":  (50, 50, 200),
        "left":  (100, 200, 100),
        "right": (180, 180, 60),
    }
    for name, colour in colours.items():
        img = Image.new("RGB", (224, 224), colour)
        img.save(cam_dir / f"{name}.png")

    rng = np.random.default_rng(seed=42)
    sensor_configs = [
        (12, ( 5, 30),  (-5,  5),   -8.0),
        ( 8, (-20, -5), (-3,  3),    0.2),
        ( 6, ( 2, 15),  ( 8, 20),   10.0),
        ( 5, ( 0, 10),  (-15, -8),   0.0),
        (10, (30, 60),  (-2,  2),    3.5),
        ( 6, ( 1,  8),  (-1,  1),    0.1),
    ]
    for i, (n, xr, yr, v_mean) in enumerate(sensor_configs):
        x = rng.uniform(*xr, size=n).astype(np.float32)
        y = rng.uniform(*yr, size=n).astype(np.float32)
        z = rng.uniform(-0.5, 2.0, size=n).astype(np.float32)
        v = rng.normal(loc=v_mean, scale=0.5, size=n).astype(np.float32)
        arr = np.stack([x, y, z, v], axis=1)
        np.save(rad_dir / f"radar_{i}.npy", arr)

    return frame_dir


# =============================================================================
# LVLM prompt builder
# =============================================================================

def build_lvlm_prompt(frame: FrameData, scenario: str = "foggy highway") -> str:
    radar_block = (
        f"<RADAR>\n{frame.radar_context}\n</RADAR>"
        if frame.radar_context
        else "<RADAR>\n[No radar data — camera-only baseline]\n</RADAR>"
    )
    prompt = f"""You are an expert autonomous driving safety evaluator.

Scenario: {scenario}
Frame ID: {frame.frame_id}

You have been provided with {len(frame.images)} camera views (front, back, left, right)
and supplementary radar sensor data below.

{radar_block}

Based on the camera imagery and the radar context above, please:
1. Identify the most critical hazards in the scene.
2. Explain your reasoning step-by-step (Graph Reasoning Chain).
3. Recommend an immediate driving action (e.g., brake, lane-change, maintain speed).
"""
    return prompt


# =============================================================================
# MiniCPM-V-2_6 inference via mlx-vlm
# =============================================================================

MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

_model_cache: dict = {}   # lazy singleton so model loads only once


def _get_model():
    """Load model + processor once, cache for reuse across ablation runs."""
    if "model" not in _model_cache:
        print(f"\n\033[94m[Model] Loading {MODEL_ID} (first run — may download weights)...\033[0m")
        import mlx.core as mx  # type: ignore[import]
        metal_ok = mx.metal.is_available()
        device   = mx.default_device()
        print(f"\033[94m[MLX]  Metal GPU available : {metal_ok}\033[0m")
        print(f"\033[94m[MLX]  Active device       : {device}\033[0m")
        if not metal_ok:
            print("\033[93m[WARN] Metal is NOT available — inference will run on CPU (slow).\033[0m")
        from mlx_vlm import load # type: ignore[import]
        from mlx_vlm.utils import load_config  # type: ignore[import]
        model, processor = load(MODEL_ID)
        config = load_config(MODEL_ID)

        # Cap max_pixels so Qwen2-VL's dynamic tiler stays inside the 32k context.
        # 448×448 = 200704 px → ~256 merged patches per image → ~256 tokens each.
        if hasattr(processor, "image_processor"):
            processor.image_processor.min_pixels = 4 * 28 * 28       # 3136
            processor.image_processor.max_pixels = 448 * 448         # 200704
            print(f"\033[94m[MLX]  Image processor max_pixels set to {448*448}\033[0m")

        _model_cache["model"] = model
        _model_cache["processor"] = processor
        _model_cache["config"] = config
        print("\033[94m[Model] Loaded ✓\033[0m")
    return _model_cache["model"], _model_cache["processor"], _model_cache["config"]


def run_model_inference(frame: FrameData, prompt: str) -> str:
    """
    Send camera images + text prompt to Qwen2-VL-2B and return the response.

    Architecture:
      1. Per-image pass: query the model once per camera with a short prompt,
         collecting a brief description for each of the 4 views.
      2. Synthesis pass: combine the 4 descriptions + radar context into a final
         safety analysis, anchored to the right-side camera image.

    Key design decision — raw prompts only:
      Do NOT call apply_chat_template before generate().  That function pre-inserts
      a single <|image_pad|> placeholder; generate()'s internal prepare_inputs then
      tries to expand it to hundreds of real patch tokens, causing a token-alignment
      mismatch and completely garbled output.  Passing the raw prompt string lets
      prepare_inputs handle templating + image token insertion in one coherent step.

    mlx-vlm's generate() requires file paths (str), not PIL Images.  We save each
    resized 448x448 image to a temp PNG so Qwen2-VL's tiler stays within the 32k
    token budget.
    """
    import tempfile, os
    from mlx_vlm import generate  # type: ignore[import]

    model, processor, _ = _get_model()

    # ── Save resized PIL images to temp files (mlx-vlm needs file paths) ──
    tmp_paths: list[str] = []
    for img in frame.images:
        resized = _resize_for_model(img)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        resized.save(path, format="PNG")
        tmp_paths.append(path)

    try:
        # ── Per-image pass: one brief description per camera view ─────────
        camera_names = ["front", "back", "left", "right"]
        per_image_descs: list[str] = []
        for img_path, cam_name in zip(tmp_paths, camera_names):
            per_img_prompt = (
                f"Briefly describe what you see in this {cam_name} camera view "
                f"of a vehicle driving in adverse weather. "
                f"Focus on hazards, road conditions, and other vehicles. "
                f"Two sentences max."
            )
            result = generate(
                model, processor,
                prompt=per_img_prompt,   # raw text — no apply_chat_template
                image=[img_path],        # list required — bare str iterates chars
                max_tokens=120,
                temperature=0.0,
                verbose=False,
                resize_shape=(448, 448),
            )
            desc = result.text if hasattr(result, "text") else str(result)
            per_image_descs.append(f"[{cam_name.upper()} camera]: {desc.strip()}")

        # ── Synthesis pass: descriptions + radar → right-camera image ─────
        # Anchor to the RIGHT camera image to avoid the broken image=None
        # text-only path in mlx-vlm/Qwen2-VL.
        radar_ctx = (
            frame.radar_context
            if frame.radar_context
            else "[No radar — camera-only baseline]"
        )
        combined_vision = "\n".join(per_image_descs)
        synthesis_prompt = (
            f"You are an expert autonomous driving safety evaluator.\n\n"
            f"Camera observations from 4 views:\n{combined_vision}\n\n"
            f"Radar sensor data:\n{radar_ctx}\n\n"
            f"Based on the above, please:\n"
            f"1. Identify the most critical hazards in the scene.\n"
            f"2. Explain your reasoning step-by-step (Graph Reasoning Chain).\n"
            f"3. Recommend an immediate driving action "
            f"(e.g., brake, lane-change, maintain speed)."
        )
        result = generate(
            model, processor,
            prompt=synthesis_prompt,     # raw text — no apply_chat_template
            image=[tmp_paths[-1]],       # list required — bare str iterates chars
            max_tokens=512,
            temperature=0.0,
            verbose=False,
            resize_shape=(448, 448),
        )
        response: str = result.text if hasattr(result, "text") else str(result)

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return response



# =============================================================================
# Pretty-print helpers
# =============================================================================

def _print_frame_report(frame: FrameData, ablation: str) -> None:
    WIDTH = 76
    label = f"Ablation: {ablation.upper()}"
    print(f"\n\033[1m── {label} {'─' * (WIDTH - len(label) - 4)}\033[0m")
    print(f"  Frame ID : {frame.frame_id}")
    print(f"  Cameras  : {len(frame.images)} images loaded  "
          f"(front · back · left · right)")

    if frame.radar_arrays:
        pts = sum(a.shape[0] for a in frame.radar_arrays)
        print(f"  Radar pts: {pts} total across {len(frame.radar_arrays)} sensors")
    else:
        print("  Radar pts: (disabled)")

    if frame.radar_context:
        ctx = frame.radar_context
        lines: list[str] = []
        while len(ctx) > 72:
            cut = ctx[:72].rfind(" ")
            cut = cut if cut > 0 else 72
            lines.append(ctx[:cut])
            ctx = ctx[cut:].lstrip()
        lines.append(ctx)
        indent = "             "
        for idx, l in enumerate(lines):
            prefix = "  Context  : " if idx == 0 else indent
            print(f"{prefix}{l}")
    else:
        print("  Context  : (empty — camera-only ablation)")

    print()


def _print_model_response(response: str, ablation: str) -> None:
    WIDTH = 76
    label = f"Model Response ({ablation.upper()})"
    print(f"\033[1m── {label} {'─' * (WIDTH - len(label) - 4)}\033[0m")
    # Word-wrap at 76 chars
    words = response.split()
    line, lines = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > 76:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    for l in lines:
        print(f"  {l}")
    print()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="StormVLM Phase I — Demo")
    ap.add_argument("--real", action="store_true",
                    help="Load real flat-file data instead of generating synthetic data")
    ap.add_argument("--baseline", action="store_true",
                    help="Run camera-only baseline (no radar context)")
    ap.add_argument("--perceive", action="store_true",
                    help="Run full application")
    ap.add_argument("--frame", default=None,
                    help="Frame ID to load (e.g. 000436). With --real, auto-detected if omitted.")
    ap.add_argument("--data-root", default=".",
                    help="Path to data root (default: current directory)")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Do not delete the synthetic temp directory after demo")
    ap.add_argument("--no-model", action="store_true",
                    help="Skip model inference — only print the prompt (faster for testing)")
    ap.add_argument("--no-radar", action="store_true",
                    help="Run without radar readings (rely solely on camera perception)")
    ap.add_argument("--evaluate", action="store_true",
                    help="Run LLM-as-a-Judge to evaluate Pipeline A (with radar) vs B (no radar)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    use_tmp = False

    # ── Data source: real vs synthetic ────────────────────────────────────
    if args.real:
        # If dataset folder exists and no root specified, use it
        if args.data_root == "." and (data_root / "dataset").is_dir():
            data_root = data_root / "dataset"
            
        frame_id = args.frame or _detect_frame_id(data_root)
        print(f"\033[94m[Demo] Real data mode — frame: {frame_id} | root: {data_root}\033[0m")
    else:
        frame_id = args.frame or "0001"
        use_tmp = True
        tmp_dir = tempfile.mkdtemp(prefix="stormvlm_demo_")
        data_root = Path(tmp_dir)
        print(f"\033[94m[Demo] Synthetic data mode — frame: {frame_id} | tmp: {data_root}\033[0m")
        _make_synthetic_dataset(data_root, frame_id=frame_id)

    # ── DriveLM-style perception mode ─────────────────────────────────────
    if args.perceive:
        from perception_node import AnchoredPerceptionNode

        print(f"\033[95m[Demo] DriveLM Perception mode — frame: {frame_id}\033[0m")
        node = AnchoredPerceptionNode()

        if args.no_model:
            # Quick test: just show annotations + prompt, no VLM
            from perception_node import AnnotationLoader, _VIEW_NAMES
            from stormvlm_loader import GlobalRadarFilter

            yaml_path = data_root / f"{frame_id}.yaml"
            annotations = AnnotationLoader.load(yaml_path)

            if args.no_radar:
                radar_anchors = []
            else:
                rf = GlobalRadarFilter()
                radar_anchors = rf.process(str(data_root), frame_id)

            view_map = AnchoredPerceptionNode._group_by_view(annotations, radar_anchors)

            print("\n" + "=" * 76)
            print(f"  DriveLM Perception — Frame {frame_id}  (--no-model)")
            print(f"  Ego speed: {annotations.get('ego_speed', 0):.1f} m/s")
            print("=" * 76)

            for view_name in _VIEW_NAMES:
                dets = view_map.get(view_name, {}).get("detections", [])
                ancs = view_map.get(view_name, {}).get("anchors", [])
                prompt = AnchoredPerceptionNode._build_prompt(view_name, dets, ancs)

                print(f"\n\033[93m[{view_name.upper()} VIEW]\033[0m")
                print(f"  Detections: {len(dets)} | Radar anchors: {len(ancs)}")
                for d in dets:
                    status = "STOPPED" if d["speed"] < 0.5 else f"{d['speed']:.1f}m/s"
                    print(f"    • {d['class']} at {d['dist']:.1f}m — {status}")
                print(f"  ── Prompt ──")
                print(f"  {prompt}")

            print("\n" + "=" * 76)
            print("\033[93m[Demo] --no-model set: skipping LVLM inference.\033[0m\n")
        else:
            from prediction_node import PredictionNode
            from planning_node import PlanningNode
            from perception_node import AnnotationLoader
            
            if args.evaluate:
                from judge_node import JudgeNode
                
                print(f"\n\033[94m[Demo] Running Pipeline A: With Radar...\033[0m")
                result_a = node.perceive_frame(data_dir=data_root, frame_id=frame_id, radar_anchors=None)
                pred_node = PredictionNode(model=node._model, processor=node._processor)
                pred_a = pred_node.predict_frame(perception_result=result_a, data_dir=data_root)
                plan_node = PlanningNode(model=node._model, processor=node._processor)
                plan_a = plan_node.plan_action(pred_a.prediction_json, result_a.ego_speed)
                
                print(f"\n\033[94m[Demo] Running Pipeline B: No Radar...\033[0m")
                result_b = node.perceive_frame(data_dir=data_root, frame_id=frame_id, radar_anchors=[])
                pred_b = pred_node.predict_frame(perception_result=result_b, data_dir=data_root)
                plan_b = plan_node.plan_action(pred_b.prediction_json, result_b.ego_speed)
                
                print(f"\n\033[95m[Demo] Running LLM-as-a-Judge Evaluation...\033[0m")
                judge = JudgeNode(model=node._model, processor=node._processor)
                
                action_a = plan_a[1] if isinstance(plan_a, tuple) else plan_a.get("selected_action", "UNKNOWN")
                reasoning_a = plan_a[0].get("planning_reasoning", "None provided.") if isinstance(plan_a, tuple) else plan_a.get("planning_reasoning", "None provided.")
                # Wait, plan_action returns a dict. Let's use get:
                action_a = plan_a.get("selected_action", "UNKNOWN")
                reasoning_a = plan_a.get("planning_reasoning", "None provided.")
                action_b = plan_b.get("selected_action", "UNKNOWN")
                reasoning_b = plan_b.get("planning_reasoning", "None provided.")
                
                pred_a_objs = len(plan_node._parse_prediction(pred_a.prediction_json))
                pred_b_objs = len(plan_node._parse_prediction(pred_b.prediction_json))
                
                weather_val = result_a.weather_type.lower()
                if "cd" in weather_val:
                    scenario_str = "clear day"
                elif "fd" in weather_val:
                    scenario_str = "heavy fog"
                elif "hd" in weather_val:
                    scenario_str = "heavy rain"
                else:
                    scenario_str = "heavy fog"
                    
                judge_res = judge.evaluate_frame(
                    frame_id=frame_id,
                    data_dir=data_root,
                    ego_speed=result_a.ego_speed,
                    weather_scenario=scenario_str,
                    pipeline_a_action=action_a,
                    pipeline_a_reasoning=reasoning_a,
                    pipeline_a_objects=pred_a_objs,
                    pipeline_b_action=action_b,
                    pipeline_b_reasoning=reasoning_b,
                    pipeline_b_objects=pred_b_objs
                )
                print("\n" + "=" * 60)
                print(f"PIPELINE A (Radar): {action_a} | {reasoning_a}")
                print(f"PIPELINE B (Camera): {action_b} | {reasoning_b}")
                print("=" * 60)
                print(judge_res.full_report())
                return
            
            radar_anchors = [] if args.no_radar else None
            result = node.perceive_frame(data_dir=data_root, frame_id=frame_id, radar_anchors=radar_anchors)
            print(result.full_report())

            pred_node = PredictionNode(model=node._model, processor=node._processor)
            pred_result = pred_node.predict_frame(perception_result=result, data_dir=data_root)
            print(pred_result.full_report())

            # Run Planning Node
            plan_node = PlanningNode(model=node._model, processor=node._processor)
            planning_result = plan_node.plan_action(pred_result.prediction_json, result.ego_speed)
            
            print("\n" + "=" * 60)
            print(f"PLANNING REPORT — Frame {frame_id}")
            print("=" * 60)
            import json
            print(json.dumps(planning_result, indent=4))
            print("\n" + "=" * 60)

        return

    # ── Original ablation modes ───────────────────────────────────────────
    modes = ["baseline"] if args.baseline else ["proposed", "baseline"]

    try:
        frames: dict[str, FrameData] = {}
        prompts: dict[str, str] = {}

        for mode in modes:
            print(f"\033[94m[Demo] Loading frame '{frame_id}' — ablation={mode}\033[0m")
            _use_radar = mode == "proposed"

            if args.real:
                # Direct flat-file loading
                frame = _load_real_frame(data_root, frame_id, use_radar=_use_radar)
                if _use_radar and frame.radar_arrays:
                    from stormvlm_loader import GlobalRadarFilter
                    fusion = GlobalRadarFilter()
                    frame.radar_context = fusion.get_radar_context(data_root, frame_id)
                else:
                    frame.radar_context = ""
            else:
                # Synthetic nested-dir loading
                loader = StormVLMLoader(data_root, use_radar=_use_radar)
                frame = loader.load_frame(frame_id)

            frames[mode] = frame
            prompts[mode] = build_lvlm_prompt(frame, scenario="foggy highway")

        # ── Reports ───────────────────────────────────────────────────────
        print("\n" + "=" * 76)
        print(f"  StormVLM Phase I — Reasoning Log   Frame: {frame_id}")
        print("=" * 76)
        for mode in modes:
            _print_frame_report(frames[mode], ablation=mode)

        # ── LVLM Prompt ───────────────────────────────────────────────────
        target_mode = "proposed" if "proposed" in frames else "baseline"
        print("=" * 76)
        print("  LVLM Prompt (what gets sent to the model)")
        print("=" * 76)
        print(prompts[target_mode])

        # ── Model inference ───────────────────────────────────────────────
        if not args.no_model:
            print("=" * 76)
            print("  Running Qwen2.5-VL-3B inference...")
            print("=" * 76)
            for mode in modes:
                response = run_model_inference(frames[mode], prompts[mode])
                _print_model_response(response, ablation=mode)
        else:
            print("\033[93m[Demo] --no-model set: skipping inference.\033[0m\n")

    finally:
        if use_tmp and not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
