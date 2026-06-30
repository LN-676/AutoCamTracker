"""GID loss benchmark manifest helpers.

The benchmark intentionally separates the small versioned manifest from the
large video files. Put videos under evaluation/gid_loss_videos and use this
module to verify the scenario set before running manual or automated passes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("evaluation") / "gid_loss_scenarios.json"


@dataclass(frozen=True)
class GIDLossScenario:
    scenario_id: str
    name: str
    video_path: Path
    description: str


@dataclass(frozen=True)
class GIDLossBenchmark:
    version: int
    video_root: Path
    metrics: dict[str, float | int]
    scenarios: list[GIDLossScenario]

    @property
    def missing_videos(self) -> list[Path]:
        return [scenario.video_path for scenario in self.scenarios if not scenario.video_path.exists()]

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scenario_count": len(self.scenarios),
            "missing_video_count": len(self.missing_videos),
            "metrics": dict(self.metrics),
            "scenarios": [
                {
                    "id": scenario.scenario_id,
                    "name": scenario.name,
                    "video": str(scenario.video_path),
                    "exists": scenario.video_path.exists(),
                }
                for scenario in self.scenarios
            ],
        }


def load_gid_loss_benchmark(manifest_path: Path | str = DEFAULT_MANIFEST) -> GIDLossBenchmark:
    manifest = Path(manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    video_root = _resolve_path(manifest.parent, Path(payload["video_root"]))
    scenarios = [
        GIDLossScenario(
            scenario_id=str(item["id"]),
            name=str(item["name"]),
            video_path=video_root / str(item["video"]),
            description=str(item.get("description", "")),
        )
        for item in payload.get("scenarios", [])
    ]
    return GIDLossBenchmark(
        version=int(payload.get("version", 1)),
        video_root=video_root,
        metrics=dict(payload.get("metrics", {})),
        scenarios=scenarios,
    )


def _resolve_path(manifest_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    project_root = manifest_dir.parent if manifest_dir.name == "evaluation" else manifest_dir
    return project_root / path

