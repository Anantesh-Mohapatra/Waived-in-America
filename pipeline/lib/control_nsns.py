"""Control-donor NSN set, derived from data already in the repo.

The control donors are exactly the items we pulled supplemental transaction
exports for: the files under raw_data/bidlink/control_contracts/. Every row
carries the NSN, so the donor set is a fact about which files are present. Some
exports record a NIIN under more than one FSC prefix (reclassifications), so we
key on the NIIN (last 9 digits) and resolve the canonical 13-digit NSN from the
DLA contract history, which is authoritative for every control item.
"""
from __future__ import annotations

import glob as _glob
from functools import lru_cache

import polars as pl

from paths import DLA_ENRICHED_LATEST, RAW_BIDLINK


def _folder_niins() -> set[str]:
    """NIINs (undashed last-9) present in the control_contracts exports."""
    niins: set[str] = set()
    for f in _glob.glob(str(RAW_BIDLINK / "control_contracts" / "*.csv")):
        for n in pl.read_csv(f, infer_schema=False)["NSN"].to_list():
            if n is None:
                continue
            u = n.replace("-", "").strip()
            if len(u) >= 9:
                niins.add(u[-9:])
    return niins


@lru_cache(maxsize=1)
def control_donor_map() -> dict[str, str]:
    """NIIN -> canonical undashed 13-digit NSN for every control donor.

    Asserts a clean 1:1 resolution: every control NIIN is present in the DLA
    contract history and maps to exactly one full NSN there."""
    niins = _folder_niins()
    dla = (
        pl.scan_parquet(DLA_ENRICHED_LATEST)
        .select(pl.col("NSN").cast(pl.Utf8))
        .with_columns(pl.col("NSN").str.replace_all("-", "").alias("u"))
        .with_columns(pl.col("u").str.slice(-9).alias("niin"))
        .filter(pl.col("niin").is_in(list(niins)))
        .group_by("niin")
        .agg(pl.col("u").unique().alias("fulls"))
        .collect(engine="streaming")
    )
    resolved = {r["niin"]: r["fulls"] for r in dla.iter_rows(named=True)}

    missing = sorted(niins - set(resolved))
    if missing:
        raise ValueError(f"{len(missing)} control NIIN(s) absent from DLA data: {missing}")
    ambiguous = {n: v for n, v in resolved.items() if len(v) != 1}
    if ambiguous:
        raise ValueError(f"{len(ambiguous)} control NIIN(s) map to >1 NSN in DLA data: {ambiguous}")

    return {niin: fulls[0] for niin, fulls in resolved.items()}


def control_donor_nsns() -> list[str]:
    """Sorted canonical undashed 13-digit NSNs of the control donors."""
    return sorted(set(control_donor_map().values()))
