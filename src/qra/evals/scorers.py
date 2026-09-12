"""Rule-based scorers. No judge in v0.1: every score is a set comparison against a label."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .template import CLASS_NAMES

OVERCLAIM_FLAGS = ["SHARPE_GT_3", "N_LT_30", "MISSING_BASIS", "CI_INCLUDES_ZERO",
                   "SUBPERIOD_WINDOW", "EXIT_DAY_MARKING"]


@dataclass(frozen=True)
class BugCatchScore:
    seeded: tuple[int, ...]
    found: tuple[int, ...]
    tp: int
    fp: int
    fn: int

    @property
    def recall(self) -> float:
        return self.tp / len(self.seeded) if self.seeded else float("nan")

    @property
    def precision(self) -> float:
        return self.tp / len(self.found) if self.found else float("nan")

    @property
    def is_control(self) -> bool:
        return not self.seeded

    @property
    def false_alarm(self) -> bool:
        return self.fp > 0


UNCLASSIFIED = 0


def score_bugcatch(seeded: list[int], found_classes: list[int]) -> BugCatchScore:
    """Class 0 = "a deviation was detected but not classified": on a seeded task it is neither a
    hit nor a false label; on a clean task it is a false alarm like any other finding."""
    s, f = set(seeded), set(found_classes)
    labelled = f - {UNCLASSIFIED}
    fp = len(labelled - s) if s else len(f)
    return BugCatchScore(tuple(sorted(s)), tuple(sorted(f)), len(s & labelled), fp, len(s - labelled))


def aggregate_bugcatch(rows: list[dict]) -> dict:
    """rows: [{"task_id", "seeded", "found", ...}] → per-class recall, precision, control FPR."""
    per_class = {}
    for k in CLASS_NAMES:
        hits = [k in r["found"] for r in rows if k in r["seeded"]]
        per_class[k] = {"name": CLASS_NAMES[k], "n": len(hits),
                        "recall": float(np.mean(hits)) if hits else float("nan")}
    seeded_rows = [r for r in rows if r["seeded"]]
    controls = [r for r in rows if not r["seeded"]]
    active = [r for r in seeded_rows if r.get("effect", "numbers") != "none"]
    dormant = [r["task_id"] for r in seeded_rows if r.get("effect") == "none"]
    by_variant = {}
    for r in seeded_rows:
        v = r.get("variant", 0)
        by_variant.setdefault(v, []).append(len(set(r["seeded"]) & set(r["found"])) / len(set(r["seeded"])))
    by_variant = {str(v): float(np.mean(x)) for v, x in sorted(by_variant.items())}
    tp = sum(len(set(r["seeded"]) & set(r["found"])) for r in rows)
    n_found = sum(len(set(r["found"]) - {UNCLASSIFIED}) for r in rows)
    n_seeded = sum(len(set(r["seeded"])) for r in rows)
    tp_active = sum(len(set(r["seeded"]) & set(r["found"])) for r in active)
    n_active = sum(len(set(r["seeded"])) for r in active)
    return {
        "per_class": per_class,
        "recall_by_variant": by_variant,
        "recall": tp / n_seeded if n_seeded else float("nan"),
        "recall_active": tp_active / n_active if n_active else float("nan"),
        "detection_active": float(np.mean([bool(r["found"]) for r in active])) if active else float("nan"),
        "dormant_tasks": dormant,
        "precision": tp / n_found if n_found else float("nan"),
        "false_labels_per_task": float(np.mean([len(set(r["found"]) - set(r["seeded"]) - {UNCLASSIFIED}) if r["seeded"] else len(set(r["found"])) for r in rows])) if rows else float("nan"),
        "control_false_alarm_rate": float(np.mean([bool(r["found"]) for r in controls])) if controls else float("nan"),
        "n_tasks": len(rows), "n_seeded_tasks": len(seeded_rows), "n_controls": len(controls),
    }


def score_overclaim(expected: list[str], got: list[str]) -> dict:
    e, g = set(expected), set(got)
    return {"exact": e == g, "missed": sorted(e - g), "extra": sorted(g - e),
            "recall": len(e & g) / len(e) if e else float("nan"),
            "over_flag": len(g - e)}


def aggregate_overclaim(rows: list[dict]) -> dict:
    per_flag = {}
    for fl in OVERCLAIM_FLAGS:
        hits = [fl in r["got"] for r in rows if fl in r["expected"]]
        per_flag[fl] = {"n": len(hits), "recall": float(np.mean(hits)) if hits else float("nan")}
    clean = [r for r in rows if not r["expected"]]
    return {
        "per_flag": per_flag,
        "exact_rate": float(np.mean([r["score"]["exact"] for r in rows])) if rows else float("nan"),
        "over_flag_per_task": float(np.mean([r["score"]["over_flag"] for r in rows])) if rows else float("nan"),
        "clean_over_flag_rate": float(np.mean([bool(r["got"]) for r in clean])) if clean else float("nan"),
        "n_tasks": len(rows),
    }
