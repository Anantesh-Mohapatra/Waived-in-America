"""Event-study model statistics shared by the table exporter and the
robustness refits.

Extracted from the table exporter so that pipeline/07_robustness/
run_event_study_refit.py can import these four helpers directly instead of
exec-loading the whole exporter module. Function bodies are verbatim.
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import REPO_ROOT  # noqa: F401  (keeps lib on sys.path for twfe_helpers)
import twfe_helpers as h  # noqa: E402

_EVENT_YEAR_COEF_RE = re.compile(r"^event_year::(-?\d+)$")


def event_year_of(name: str) -> int | None:
    m = _EVENT_YEAR_COEF_RE.match(str(name))
    return int(m.group(1)) if m else None


def joint_pretrend_pvalue(model) -> float | None:
    """Joint Wald F-test that all pre-period (k<-1) event_year coefs = 0.

    pyfixest's wald_test forces chi^2 when R is non-identity (issuing a warning)
    and stores chi^2-based ``_p_value``. We want the F-distribution p-value
    with (dfn, dfd), which is the convention for paper-reported pre-trend
    tests, so we reconstruct it from ``_f_statistic`` directly.

    Returns the p-value, or None if there are no pre-period coefs in the model.
    """
    coefs = model.coef()
    names = list(coefs.index)
    pre_idx = [
        i for i, n in enumerate(names)
        if (ey := event_year_of(n)) is not None and ey < -1
    ]
    if not pre_idx:
        return None
    k = len(coefs)
    R = np.zeros((len(pre_idx), k))
    for row, col in enumerate(pre_idx):
        R[row, col] = 1.0
    q = np.zeros(len(pre_idx))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.wald_test(R=R, q=q, distribution="F")
    fstat = float(model._f_statistic)
    dfn = int(model._dfn)
    dfd = int(model._dfd)
    return float(fdist.sf(fstat, dfn, dfd))


def avg_post_beta(model) -> tuple[float, float] | tuple[None, None]:
    """Linear combination beta_bar = mean(beta_k for k>=0, k != sentinel)
    with CRV1 SE from the model's stored vcov."""
    coefs = model.coef()
    names = list(coefs.index)
    post_idx = [
        i for i, n in enumerate(names)
        if (ey := event_year_of(n)) is not None
        and ey >= 0 and ey != h.EVENT_YEAR_SENTINEL
    ]
    if not post_idx:
        return None, None
    k = len(coefs)
    R = np.zeros((1, k))
    weight = 1.0 / len(post_idx)
    for col in post_idx:
        R[0, col] = weight
    b = coefs.values
    V = model._vcov
    est = float((R @ b)[0])
    se = float(np.sqrt((R @ V @ R.T)[0, 0]))
    return est, se


def n_clusters(model) -> int:
    g = getattr(model, "_G", None)
    if g is None:
        return 0
    if isinstance(g, (list, tuple, np.ndarray)):
        return int(g[0]) if len(g) else 0
    return int(g)


# ---------- event_study.json -> thesis-generator shapes ----------
# Single owner of the two transforms that turn an event_study.json outcome
# block into the dict shapes the thesis figure/table generators consume. Used
# by both the main thesis generators (build_thesis_{figures,tables}) and the
# per-variant builder (07_robustness/build_artifacts), so main and variants
# render from one code path.

def es_spec_from_json(es: dict) -> dict:
    """JSON `es_coefs` block -> ES_COEFS[outcome] shape (figure coefplot, spec 3).
    The coefplot marks the omitted reference period, so it expects a (-1, 0, 0)
    row; the JSON omits the reference (no coef estimated), so inject it."""
    coefs = [tuple(c) for c in es["coefs"]]
    if not any(c[0] == -1 for c in coefs):
        coefs.append((-1, 0.0, 0.0))
    coefs.sort(key=lambda c: c[0])
    return {
        "title": es["title"],
        "ylabel": es["ylabel"],
        "n_obs": es["n_obs"],
        "n_clusters": es["n_clusters"],
        "pretrend_p": es["pretrend_p"],
        "avg_post": tuple(es["avg_post"]),
        "coefs": coefs,
        "n_treated_per_ey": {int(ey): int(n) for ey, n in es["n_treated_per_ey"]},
    }


def ladder_from_json(ld: dict) -> dict:
    """JSON `ladder` block -> LADDER_DATA[outcome] shape (appendix 3-spec ladder)."""
    return {
        "title": ld["title"],
        "coefs": {int(ey): [tuple(cell) for cell in specs] for ey, specs in ld["coefs"]},
        "pretrend_p": [tuple(x) for x in ld["pretrend_p"]],
        "avg_post": [tuple(x) for x in ld["avg_post"]],
        "n_obs": ld["n_obs"],
        "r2": ld["r2"],
    }
