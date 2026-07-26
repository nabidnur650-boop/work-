from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "q1_90/scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_range_partition_is_exact() -> None:
    download = load("parallel_range_download")
    assert [download.part_bounds(index, 10, 37) for index in range(4)] == [
        (0, 9),
        (10, 19),
        (20, 29),
        (30, 36),
    ]


def test_deterministic_event_placement_respects_exclusion_buffer() -> None:
    prepare = load("prepare_external_recipes")
    target = (40.0, 44.0)
    first = prepare.distractor_starts(180, target, [3.0, 4.0, 2.0], "fixed")
    second = prepare.distractor_starts(180, target, [3.0, 4.0, 2.0], "fixed")
    assert first == second
    intervals = [(target[0] - 1.0, target[1] + 1.0)]
    for start, duration in zip(first, [3.0, 4.0, 2.0], strict=True):
        current = (start - 1.0, start + duration + 1.0)
        assert all(current[1] <= old[0] or current[0] >= old[1] for old in intervals)
        intervals.append(current)


def test_external_bootstrap_reproduces_constant_paired_effect() -> None:
    evaluate = load("evaluate_external_replication")
    rows = []
    for class_id in range(4):
        for source in range(3):
            rows.append(
                {
                    "class_id": class_id,
                    "foreground_cluster_id": f"{class_id}-{source}",
                    "ensemble": {"evidence_ap": 0.8, "hit_at_1": 1.0},
                    "prior": {"evidence_ap": 0.5, "hit_at_1": 0.0},
                }
            )
    interval = evaluate.hierarchical_bootstrap(rows, seed=11, replicates=100)
    assert np.allclose(interval["evidence_ap"], [0.3, 0.3])
    assert interval["hit_at_1"] == [1.0, 1.0]


def test_model_blind_amendment_matches_frozen_config() -> None:
    config = json.loads(
        (PROJECT / "q1_90/configs/external_replication.json").read_text(encoding="utf-8")
    )
    amendment = json.loads(
        (PROJECT / config["model_blind_amendment"]).read_text(encoding="utf-8")
    )
    assert amendment["performance_information_used"] is False
    assert amendment["trigger"]["recipes_written"] == 0
    assert amendment["trigger"]["model_scores_computed"] == 0
    assert amendment["amended_eligible_folds"] == config["foreground"]["eligible_folds"]
    assert min(amendment["amended_esc_disjoint_unique_sources_by_class"].values()) >= 12
    assert min(amendment["original_esc_disjoint_unique_sources_by_class"].values()) < 12
