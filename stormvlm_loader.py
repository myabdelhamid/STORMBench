"""
StormVLM — Phase I: Multi-Modal Loader Engine
================================================
Master's Thesis · GIU Berlin
Author: Marwan Elsayed

Hardware Target: MacBook Pro M1 · 8 GB RAM
Inference Backend: mlx-vlm (4-bit quantised, Apple Silicon)

Architecture Overview
---------------
StormVLMLoader
  └── load_frame(frame_id)
        ├── _load_cameras()   → list[PIL.Image]
        └── _load_radars()    → list[np.memmap]  (zero-copy, stays on disk)

GlobalRadarFilter  (data reduction only — no heuristic judgment)
  └── process(data_dir, frame_id) → list[dict]
        ├── _load_and_fuse()  → (N, 5) [x, y, z, v, sensor_id]
        ├── _gate()           → spatiotemporal filter (range, Doppler, Z)
        ├── _cluster()        → DBSCAN on 2-D [x, y]
        └── _extract_anchors()→ centroid + speed + camera mapping per cluster
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np # type: ignore[import]
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[StormVLM %(levelname)s] %(message)s",
)
log = logging.getLogger("stormvlm")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class FrameData:
    """Holds all raw modalities for a single timestamp / frame."""

    frame_id: str

    # 4 camera images: [front, back, left, right]
    images: List[Image.Image] = field(default_factory=list)

    # 6 radar point clouds (memory-mapped, shape N×4 each)
    # Columns: [x, y, z, doppler_velocity]
    radar_arrays: List[np.ndarray] = field(default_factory=list)

    # Natural-language radar summary (populated by GlobalRadarFilter)
    radar_context: str = ""

    @property
    def num_points(self) -> int:
        """Total lidar/radar points across all sensors."""
        return sum(r.shape[0] for r in self.radar_arrays if r is not None)


# ---------------------------------------------------------------------------
# StormVLMLoader
# ---------------------------------------------------------------------------
class StormVLMLoader:
    """
    Loads 4-camera and 6-radar data for a given frame from disk.

    Parameters
    ----------
    data_root : str | Path
        Root directory containing per-frame sub-folders.
    use_radar : bool
        If False, radar files are skipped entirely (camera-only baseline).
    camera_dirs : list[str]
        Sub-folder names for each camera inside frame directories.
        Default: ["front", "back", "left", "right"]
    radar_count : int
        Number of radar sensors (default 6, indexed 0-5).
    """

    CAMERA_NAMES = ["front", "back", "left", "right"]

    def __init__(
        self,
        data_root: str | Path,
        use_radar: bool = True,
        camera_dirs: Optional[List[str]] = None,
        radar_count: int = 6,
    ) -> None:
        self.data_root = Path(data_root)
        self.use_radar = use_radar
        self.camera_dirs = camera_dirs or self.CAMERA_NAMES
        self.radar_count = radar_count

        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        log.info(
            "StormVLMLoader initialised — radar=%s | root=%s",
            self.use_radar,
            self.data_root,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_frame(self, frame_id: str) -> FrameData:
        """
        Load all modalities for *frame_id*.

        Expected on-disk layout
        -----------------------
        <data_root>/
          <frame_id>/
            cameras/
              front.jpg   (or .png)
              back.jpg
              left.jpg
              right.jpg
            radars/
              radar_0.npy
              radar_1.npy
              ...
              radar_5.npy
        """
        frame_dir = self.data_root / frame_id
        if not frame_dir.exists():
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

        frame = FrameData(frame_id=frame_id)
        frame.images = self._load_cameras(frame_dir)

        if self.use_radar:
            frame.radar_arrays = self._load_radars(frame_dir)
            log.info(
                "Frame '%s' loaded — cameras=%d  radar_points=%d",
                frame_id,
                len(frame.images),
                frame.num_points,
            )
        else:
            log.info(
                "Frame '%s' loaded — cameras=%d  (radar disabled)",
                frame_id,
                len(frame.images),
            )

        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_cameras(self, frame_dir: Path) -> List[Image.Image]:
        """Load 4 camera images; returns PIL Images (RGB)."""
        cam_dir = frame_dir / "cameras"
        images: List[Image.Image] = []

        for cam_name in self.camera_dirs:
            img_path = self._find_image(cam_dir, cam_name)
            if img_path is None:
                log.warning("Camera image not found for '%s' in %s", cam_name, cam_dir)
                images.append(Image.new("RGB", (224, 224)))  # black placeholder
            else:
                images.append(Image.open(img_path).convert("RGB"))

        return images

    def _load_radars(self, frame_dir: Path) -> List[np.ndarray]:
        """
        Load 6 radar .npy files using memory-mapped I/O.

        np.load(..., mmap_mode='r') maps the file into virtual address space
        without copying data into RAM — critical for the 8 GB M1 constraint.
        Each returned array has shape (N, 4): [x, y, z, doppler_velocity].
        """
        radar_dir = frame_dir / "radars"
        arrays: List[np.ndarray] = []

        for i in range(self.radar_count):
            path = radar_dir / f"radar_{i}.npy"
            if not path.exists():
                log.warning("Radar file missing: %s — using empty array", path)
                arrays.append(np.empty((0, 4), dtype=np.float32))
            else:
                # mmap_mode='r'  → read-only, zero-copy, stays on disk
                arr = np.load(str(path), mmap_mode="r")
                if arr.ndim != 2 or arr.shape[1] != 4:
                    log.error(
                        "Unexpected radar shape %s in %s — skipping", arr.shape, path
                    )
                    arrays.append(np.empty((0, 4), dtype=np.float32))
                else:
                    arrays.append(arr)

        return arrays

    @staticmethod
    def _find_image(cam_dir: Path, stem: str) -> Optional[Path]:
        """Return the first image file matching *stem* with any extension."""
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            p = cam_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None


# ---------------------------------------------------------------------------
# GlobalRadarFilter — Slim Data-Reduction Engine (Option C)
# ---------------------------------------------------------------------------
# Design philosophy:
#   This module performs ONLY mathematically objective operations on radar data.
#   All interpretation (object class, visibility, motion intent) is deferred
#   to the LVLM reasoning layer, which can cross-reference with camera images.
#
# Academic references:
#   RadarLLM (Yan et al., 2024) — "Radar-to-Text" anchor token concept.
#   SlimComm dataset (2024) — 6-radar 360° sensor cocoon structure.
# ---------------------------------------------------------------------------
class GlobalRadarFilter:
    """
    Fuses 6 SlimComm radar point clouds into a sparse set of DBSCAN-clustered
    "Anchor" objects and outputs neutral data for LVLM reasoning.

    Pipeline (data reduction only — no heuristic judgment)
    ------
    Stage 1  Load & Fuse
             └── Concatenate 6 radar files → (N, 5) [x, y, z, v, sensor_id]

    Stage 2  Spatiotemporal Gating  (vectorised NumPy)
             ├── Range gate   : √(x²+y²) < range_max_m           (50 m)
             ├── Doppler gate : |v| ≥ doppler_min_ms              (2.0 m/s)
             └── Z-filter     : |z| ≤ z_max_m                     (3.0 m)

    Stage 3  Spatial Clustering
             └── sklearn DBSCAN on 2-D [x, y]

    Stage 4  Anchor Extraction & Camera Mapping
             ├── Centroid (x̄, ȳ) per cluster
             ├── Max |Doppler| and mean signed Doppler per cluster
             └── Camera-view label via hard-specified mapping table:

               ┌──────────┬──────────────┬──────────────────┬────────────────────┐
               │ Radar    │ Position     │ Primary Camera   │ Secondary Camera   │
               ├──────────┼──────────────┼──────────────────┼────────────────────┤
               │   0      │ Front        │ Camera 0 (Front) │ —                  │
               │   1      │ Front-Right  │ Camera 0 (Front) │ Camera 1 (Right)   │
               │   2      │ Front-Left   │ Camera 0 (Front) │ Camera 2 (Left)    │
               │   3      │ Back         │ Camera 3 (Back)  │ —                  │
               │   4      │ Back-Left    │ Camera 3 (Back)  │ Camera 2 (Left)    │
               │   5      │ Back-Right   │ Camera 3 (Back)  │ Camera 1 (Right)   │
               └──────────┴──────────────┴──────────────────┴────────────────────┘

    Output: list[dict] with neutral, objective fields only.
    The LVLM is responsible for classification, visibility assessment,
    deduplication, and motion intent reasoning using both anchors + camera images.
    """

    # Hard-specified mapping table (user-confirmed 2026-03-06)
    _SENSOR_CAMERA_MAP: dict[int, dict[str, str]] = {
        0: {"position": "Front",       "primary": "Camera 0 (Front)", "secondary": ""},
        1: {"position": "Front-Right", "primary": "Camera 0 (Front)", "secondary": "Camera 1 (Right)"},
        2: {"position": "Front-Left",  "primary": "Camera 0 (Front)", "secondary": "Camera 2 (Left)"},
        3: {"position": "Back",        "primary": "Camera 3 (Back)",  "secondary": ""},
        4: {"position": "Back-Left",   "primary": "Camera 3 (Back)",  "secondary": "Camera 2 (Left)"},
        5: {"position": "Back-Right",  "primary": "Camera 3 (Back)",  "secondary": "Camera 1 (Right)"},
    }

    def __init__(
        self,
        range_max_m: float = 50.0,
        doppler_min_ms: float = 0.0,
        z_max_m: float = 3.0,
        dbscan_eps: float = 2.5,
        dbscan_min_pts: int = 3,
    ) -> None:
        self.range_max_m = range_max_m
        self.doppler_min_ms = doppler_min_ms
        self.z_max_m = z_max_m
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_pts = dbscan_min_pts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, data_dir: str | Path, frame_id: str) -> list[dict]:
        """
        Full pipeline: load → gate → cluster → extract anchors.

        Returns
        -------
        list[dict] — one dict per anchor with ONLY objective fields:
            id, x, y, dist_m, max_speed_ms, mean_doppler_ms,
            n_points, dominant_sensor, sensor_position,
            primary_camera, secondary_camera
        """
        cloud = self._load_and_fuse(data_dir, frame_id)

        if cloud.shape[0] == 0:
            log.warning("[GlobalRadarFilter] Empty point cloud after fusion.")
            return []

        gated = self._gate(cloud)

        if gated.shape[0] < self.dbscan_min_pts:
            log.info(
                "[GlobalRadarFilter] %d points after gating — below min_samples.",
                gated.shape[0],
            )
            return []

        labels  = self._cluster(gated)
        anchors = self._extract_anchors(gated, labels)

        log.info(
            "[GlobalRadarFilter] %d anchors from %d gated points (of %d raw).",
            len(anchors), gated.shape[0], cloud.shape[0],
        )
        return anchors

    def get_anchor_tokens(self, data_dir: str | Path, frame_id: str) -> list[str]:
        """
        Return one neutral data token string per anchor.
        No interpretation — just measurements for LVLM consumption.

        Format:
          "Anchor [id]: [n_points] pts at [dist]m ([x],[y]),
           max_speed=[speed]m/s mean_doppler=[doppler]m/s,
           sensor=[position] → [primary] / [secondary]"
        """
        anchors = self.process(data_dir, frame_id)
        tokens: list[str] = []
        for a in anchors:
            secondary = f" / {a['secondary_camera']}" if a["secondary_camera"] else ""
            token = (
                f"Anchor {a['id']}: {a['n_points']} radar pts at "
                f"{a['dist_m']:.1f}m ({a['x']:+.1f}, {a['y']:+.1f}), "
                f"max_speed={a['max_speed_ms']:.1f}m/s "
                f"mean_doppler={a['mean_doppler_ms']:+.1f}m/s, "
                f"sensor={a['sensor_position']} → {a['primary_camera']}{secondary}"
            )
            tokens.append(token)
        return tokens

    def get_radar_context(self, data_dir: str | Path, frame_id: str) -> str:
        """
        Single string for direct injection into an LVLM prompt.
        Provides raw measurements; the LVLM decides what they mean.
        """
        tokens = self.get_anchor_tokens(data_dir, frame_id)
        if not tokens:
            return (
                "Radar: No dynamic objects detected within "
                f"{self.range_max_m:.0f}m (Doppler ≥ {self.doppler_min_ms} m/s). "
                "Note: stationary objects (stopped cars, barriers) are filtered out."
            )
        header = (
            f"Radar ({len(tokens)} clusters, Doppler ≥ {self.doppler_min_ms} m/s, "
            f"range < {self.range_max_m:.0f}m). "
            "Note: stationary objects are NOT included — check cameras."
        )
        return header + "\n" + "\n".join(f"  • {t}" for t in tokens)

    # ------------------------------------------------------------------
    # Stage 1 — Load & Fuse
    # ------------------------------------------------------------------
    def _load_and_fuse(self, data_dir: str | Path, frame_id: str) -> np.ndarray:
        """
        Load 6 radar .npy files, tag each point with sensor_id (col 4),
        concatenate into one global point cloud.

        Output shape: (N_total, 5) — columns [x, y, z, v, sensor_id]
        """
        root = Path(data_dir)
        arrays: list[np.ndarray] = []

        for i in range(6):
            candidates = [
                root / f"{frame_id}_radar{i}.npy",
                root / frame_id / "radars" / f"radar_{i}.npy",
            ]
            for path in candidates:
                if path.exists():
                    arr = np.load(str(path), mmap_mode="r")
                    if arr.ndim == 2 and arr.shape[1] >= 4:
                        pts = np.asarray(arr[:, :4], dtype=np.float32)
                        sid = np.full((pts.shape[0], 1), i, dtype=np.float32)
                        arrays.append(np.concatenate([pts, sid], axis=1))
                    break

        if not arrays:
            return np.empty((0, 5), dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    # ------------------------------------------------------------------
    # Stage 2 — Gating (vectorised, no Python loops)
    # ------------------------------------------------------------------
    def _gate(self, cloud: np.ndarray) -> np.ndarray:
        """Apply range + Doppler + Z gates. Columns: [x, y, z, v, sensor_id]."""
        x, y, z, v = cloud[:, 0], cloud[:, 1], cloud[:, 2], cloud[:, 3]
        mask = (
            (np.hypot(x, y) < self.range_max_m)
            & (np.abs(v) >= self.doppler_min_ms)
            & (np.abs(z) <= self.z_max_m)
        )
        return cloud[mask]

    # ------------------------------------------------------------------
    # Stage 3 — Clustering
    # ------------------------------------------------------------------
    def _cluster(self, gated: np.ndarray) -> np.ndarray:
        """DBSCAN on 2-D [x, y]. Returns per-point cluster labels."""
        try:
            from sklearn.cluster import DBSCAN  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "scikit-learn required: pip install scikit-learn"
            ) from exc

        return DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_pts,
            algorithm="ball_tree",
            n_jobs=1,
        ).fit_predict(gated[:, :2])

    # ------------------------------------------------------------------
    # Stage 4 — Anchor Extraction (neutral data only)
    # ------------------------------------------------------------------
    def _extract_anchors(
        self, gated: np.ndarray, labels: np.ndarray
    ) -> list[dict]:
        """
        Per cluster: centroid, max speed, mean Doppler, point count,
        dominant sensor → camera mapping. No classification or interpretation.
        """
        anchors: list[dict] = []

        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue

            pts = gated[labels == cluster_id]

            x_c = float(pts[:, 0].mean())
            y_c = float(pts[:, 1].mean())

            # Majority-vote sensor → camera mapping
            sensor_ids = pts[:, 4].astype(np.int32)
            dominant = int(np.bincount(sensor_ids).argmax())
            mapping = self._SENSOR_CAMERA_MAP.get(
                dominant, {"position": "Unknown", "primary": "Unknown", "secondary": ""}
            )

            anchors.append({
                "id":                int(cluster_id),
                "x":                 x_c,
                "y":                 y_c,
                "dist_m":            float(np.hypot(x_c, y_c)),
                "max_speed_ms":      float(np.abs(pts[:, 3]).max()),
                "mean_doppler_ms":   float(pts[:, 3].mean()),
                "n_points":          int(pts.shape[0]),
                "dominant_sensor":   dominant,
                "sensor_position":   mapping["position"],
                "primary_camera":    mapping["primary"],
                "secondary_camera":  mapping["secondary"],
            })

        anchors.sort(key=lambda a: a["dist_m"])
        return anchors

