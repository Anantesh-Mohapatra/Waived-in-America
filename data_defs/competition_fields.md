# Competition Fields Reference (USAspending / FPDS)

**Source dataset:** `data/clean/procurement_data.parquet` — the cleaned USAspending download. This is **the** source for all procurement competition data in the project. Downstream files (`dla_contract_enriched_*.parquet`, `dla_contract_history.parquet`, etc.) inherit these columns from here; if their values differ, the difference was introduced by a downstream cleaning or join step, not by FPDS.

**Parquet scope:** `award_or_idv_flag = 'AWARD'` on 100% of rows — the parquet contains no IDV records. Any field that is IDV-only (notably `multiple_or_single_award_idv`) will be 100% null here by construction, not because FPDS omits it.

**Column naming convention:** every categorical has a `_code` column (stable) and a human-readable twin. **Always join/filter on the code.** Description text was reworded in the USAspending download mid-FY2022 (e.g. "COMMERCIAL ITEM" → "COMMERCIAL PRODUCTS/SERVICES"); FY2022 has both strings, FY2023+ has only the new wording. The codes themselves did not change.

**Authoritative source:** [FPDS-NG Data Dictionary v1.4](https://web.archive.org/web/20251105182236/https://www.fpds.gov/wiki/index.php/V1.4_FPDS-NG_Data_Dictionary). FPDS links were broken when the system moved to sam.gov; the link above is the web.archive.org snapshot. All code definitions, requirement-state rules ("base vs mod"), and short descriptions in this reference were pulled verbatim from that page. Codes observed in the parquet but **not** in v1.4 (e.g. `WOSB`, `EDWOSB`, `ISBEE`, commercial-item `E`) are flagged as post-v1.4 additions.

**A universal rule from the dictionary — mods almost never report competition fields directly.** For every field below (`extent_competed`, `type_of_set_aside`, `number_of_offers_received`, `solicitation_procedures`, `other_than_full_and_open_competition`, `commercial_item_acquisition_procedures`, `fair_opportunity_limited_sources`), the dictionary's "Mod" row is `N` across all contract types. In FPDS these fields are collected on the base award only; modifications inherit by reference. USAspending, however, propagates the base value to each mod row in its export, so in this parquet you will typically see the field populated on mods too — but understand that you are looking at the base award's value, not an independent reassessment.

---

## `extent_competed_code` / `extent_competed`  (element 10A)

**Definition (FPDS):** "A code that represents the competitive nature of the contract." (FAR 4.601(c)(1); DFARS 253.204-70(c)(4)(iii)(vi))

**Null rate:** ≈0% in the raw parquet (278 nulls). This is because USAspending propagates the base value onto mods — per the dictionary, mods themselves are not required to report this field.

| Code | FPDS short description | Competed? | Rows in parquet |
|---|---|---|---|
| A | Full and Open Competition | Yes | 24.13M |
| F | Competed under SAP | Yes | 5.53M |
| C | Not Competed | No | 2.01M |
| B | Not Available for Competition | No | 2.00M |
| D | Full and Open Competition after exclusion of sources | Yes | 1.71M |
| G | Not Competed under SAP | No | 1.25M |
| CDO | Competitive Delivery Order | Yes | 2,031 |
| NDO | Non-Competitive Delivery Order | No | 316 |
| E | Follow On to Competed Action | — | 110 |

SAP = Simplified Acquisition Procedures, as defined in FAR 13. The dictionary does **not** state a dollar threshold for SAP — the threshold is set separately in FAR 2.101/13 and changes over time. Look up the current threshold there if you need it; do not hardcode a number in analysis.

**Per-row from the v1.4 dictionary:**
- **A**: "Report this code if the action resulted from an award pursuant to FAR 6.102(a) — sealed bid, FAR 6.102(b) — competitive proposal, FAR 6.102(c) — Combination, or any other competitive method that did not exclude sources of any type."
- **B**: "Select this code when the contract is not available for competition."
- **C**: "Select this code when the contract is not competed."
- **D**: "Select this code when some sources are excluded before competition."
- **F**: "Select this code when the action is competed under the Simplified Acquisition Threshold."
- **G**: "Select this code when the action is NOT competed under the Simplified Acquisition Threshold."
- **E**: "Select this code when the action is a follow on to an existing competed contract. FAR 6.302-1."
- **CDO**: "Report this code when the IDV Type is a Federal Schedule … delivery/task order award was made pursuant to a process that permitted each contract awardee a fair opportunity to be considered. See FAR Part 16.505(b)(1)."
- **NDO**: "Report this code when competitive procedures are not used in awarding the delivery order for a reason not included above."

**Binary rule (covers > 99.99% of rows):**
```
competed     = code in {'A','D','F','CDO'}
not_competed = code in {'B','C','G','NDO'}
```
Treat `E` (110 rows) as a judgment call and ignore for aggregate analysis.

---

## `type_of_set_aside_code` / `type_of_set_aside`  (element 10N)

**Definition (FPDS):** "The designator for type of set aside determined for the contract action." (FAR 19.502-2, 19.805-2, 19.502-3, DFARS 226.7003, 235.016, FAR 19.9–19.14, FAR 13)

**Critical rule:** `null ≠ "no set-aside"`. The explicit "no set-aside" value is code `NONE` ("No set aside used"). **Always pool `null` and `NONE` as "not set aside"** — otherwise FY2018+ data looks artificially depleted (see observed pattern below).

### Codes documented in v1.4
| Code | FPDS short description |
|---|---|
| NONE | No set aside used |
| SBA | Small Business Set Aside - Total |
| SBP | Small Business Set-Aside – Partial |
| 8A | 8A Competed |
| 8AN | 8(a) Sole Source |
| 8AC | SDB Set-Aside 8(a) (deprecated) |
| SDVOSBC | Service Disabled Veteran Owned Small Business Set-Aside |
| SDVOSBS | SDVOSB Sole Source |
| HZC | HUBZone Set-Aside |
| HZS | HUBZone Sole Source |
| HS2 | Combination HUBZone and 8(a) (deprecated) |
| HS3 | 8(a) with HUB Zone Preference |
| HMP | HBCU or MI Set-Aside – Partial |
| HMT | HBCU or MI Set-Aside – Total |
| VSA | Veteran Set Aside |
| VSS | Veteran Sole Source |
| ESB | Emerging Small Business Set-Aside |
| RSB | Reserved for Small Business $2,501 to $100K (pre-CLOD only) |
| VSB | Very Small Business Set Aside |
| BI | Buy Indian (Interior / HHS-IHS only) |

### Codes observed in our parquet but NOT in v1.4
These appear in the data and are legitimate — they were added to FPDS after v1.4 was published. Descriptions here are NOT sourced from the v1.4 dictionary:

| Code | Meaning (not v1.4-sourced) |
|---|---|
| WOSB | Women-Owned Small Business |
| WOSBSS | WOSB Sole Source |
| EDWOSB | Economically Disadvantaged WOSB |
| EDWOSBSS | EDWOSB Sole Source |
| ISBEE | Indian Small Business Economic Enterprise |

### Observed null pattern (empirical, not documented)
FY2017 rows are 2.6% null with 89% literal `NONE`. Starting FY2018 that flips: ~76% null, only ~18% `NONE`. The non-`NONE` codes (SBA, SDVOSBC, HZC, 8AN, etc.) have nearly identical raw counts across the break — it is specifically the `NONE` → null conversion that changes.

Cause is **not independently confirmed**. The v1.4 dictionary's Requirement State rows (Base = R for most contract types, Mod = N everywhere) have not changed, so this is most likely a USAspending export/schema change at the FY2017/FY2018 boundary, not an FPDS reporting-rule change. Treat it as observed-in-our-parquet, not a documented FPDS behavior.

---

## `number_of_offers_received`  (element 10D)

**Definition (FPDS):** "The number of actual offers/bids received in response to the solicitation." (FAR 4.601(b)(5), 4.601(d)(3); DFARS 253.204-70(c)(4)(vii))

**Stored as a string in the parquet — cast before comparing:**
```python
pl.col("number_of_offers_received").cast(pl.Int64, strict=False)
```

### Why nulls are so common (67% overall)
Per the v1.4 dictionary, `Mod = N` across **all** contract types in both civilian and DoD, pre-CLOD and post-CLOD. In other words, modifications are explicitly NOT required to report this field, by FPDS rule. Even on base awards, the requirement varies: Delivery Orders off single-award IDVs are `P` (derived), FSS/GWAC/BOA/BPA bases are `N` (not reported), and only DO-Multiple, PO>25K, DCA, and IDC bases are `R` (required). The null pattern in the parquet is therefore the direct product of documented FPDS rules, not a data-quality issue.

Zeros effectively never appear (2,892 rows). Null is not zero.

### Null rate and median by `extent_competed_code`
| Code | Null rate | Median (non-null) |
|---|---|---|
| A | 82% | 6 |
| B | 97% | 1 |
| C | 71% | 1 |
| D | 65% | 5 |
| F | 7% | 3 |
| G | 3% | 1 |

### Practical rules
- **Don't treat this as a row-level competition feature.** Most rows don't carry a real count.
- **Don't median across codes.** The distribution is extremely heavy-tailed on multi-award IDVs (e.g. D). Scope filters can move the median by 5× or more.
- For a robust competition signal, use `extent_competed_code` binary instead.
- If you need offer counts, restrict to `F`/`G` where coverage is > 90%, or roll up to parent-contract level.

---

## `solicitation_procedures_code` / `solicitation_procedures`  (element 10M)

**Definition (FPDS):** "The designator for competitive solicitation procedures available." (FAR 4.601(c), 6, 8.404, 13.003, 16.505, 19; DFARS 204.670, 226.7003, 253.204-70)

**Derivation note (from dictionary):** "Derived for Delivery Order against a Multiple Award GWAC, FSS, or IDC with 'Subject to Multiple Award Fair Opportunity'." That is why `MAFO` shows up so heavily on DOs — it's auto-set by FPDS, not manually entered.

| Code | FPDS short description |
|---|---|
| AE | Architect-Engineer FAR 6.102 |
| AS | Alternative Sources |
| BR | Basic Research |
| MAFO | Subject to Multiple Award Fair Opportunity |
| NP | Negotiated Proposal/Quote |
| NONE | No solicitation procedure used (civilian-only) |
| SB | Sealed Bid |
| SP1 | Simplified Acquisition |
| SSS | Single Source Solicited (DoD-only) |
| TS | Two Step |

For NP/SB/TS, the dictionary notes "contract award over $100K using [these] procedures" — those thresholds are from FAR 12/13/14/15 and may have drifted since v1.4.

---

## `other_than_full_and_open_competition_code` / `other_than_full_and_open_competition`  (element 10C)

**Definition (FPDS):** "The designator for solicitation procedures other than full and open competition pursuant to FAR 6.3."

**Populated only when `extent_competed_code` is not in {A, D, F}** (i.e. sole-source or SAP-non-competed categories). In the parquet, 85% of rows are null for this field by design. Mods are `N` for this field across the board.

### All v1.4 codes
| Code | FPDS short description |
|---|---|
| OTH | Authorized by Statute (FAR 6.302-5(a)(2)(i)) |
| RES | Authorized Resale (FAR 6.302-5(a)(2)(ii)) |
| BND | Brand Name Description (FAR 6.302-1(c)) |
| FOC | Follow-On Contract (FAR 6.302-1(a)(2)(ii/iii)) |
| IA | International Agreement (FAR 6.302-4) |
| MPT | Less than or equal to the Micro-Purchase Threshold (civilian) |
| MES | Mobilization, Essential R&D (FAR 6.302-3) |
| NS | National Security (FAR 6.302-6) |
| ONE | Only One Source - Other (FAR 6.302-1 other) |
| PDR | Patent or Data Rights (FAR 6.302-1(b)(2)) |
| PI | Public Interest (FAR 6.302-7) |
| SP2 | SAP Non-Competition (FAR 13) |
| STD | Standardization (FAR 6.302-1(b)(4)) |
| UNQ | Unique Source (FAR 6.302-1(b)(1)) |
| UR | Unsolicited Research Proposal (FAR 6.302-1(a)(2)(i)) |
| URG | Urgency (FAR 6.302-2) |
| UT | Utilities (FAR 6.302-1(a)(2) & (b)(3)) |

---

## `fair_opportunity_limited_sources_code` / `fair_opportunity_limited_sources`  (element 10R)

**Definition (FPDS):** "The type of statutory exception to Fair Opportunity." Instruction: "Report this code when awarding a non competitive task order or delivery order exceeding $2500.00 against an IDIQ contract."

**Populated scope (from v1.4 Requirement State):** **only `DO Multiple` has `R`** (required). Every other contract type is `NA` (not applicable) or `N`. In plain terms: **this field only appears on task/delivery orders issued against multiple-award IDVs.** The ~19% non-null rate I observed earlier is consistent with that.

### v1.4 codes
| Code | FPDS short description |
|---|---|
| FAIR | Fair Opportunity Given |
| FOO | Follow-on Delivery Order Following Competitive Initial Order (FAR 16.505(B)(2)(iii)) |
| MG | Minimum Guarantee (FAR 16.505(b)(2)(iv)) |
| ONE | Only One Source - Other (FAR 16.505(B)(2)(ii)) |
| OSA | Other Statutory Authority |
| URG | Urgency (FAR 16.505(B)(2)(i)) |

The codes `CSA`, `SS`, `LSRC`, `SSRC` observed in the parquet are **not** in v1.4 — they are post-v1.4 additions.

---

## `commercial_item_acquisition_procedures_code` / `commercial_item_acquisition_procedures`  (element 10H)

**Definition (FPDS):** "Designates whether the solicitation used the special requirements for the acquisition of commercial items … as defined by FAR Part 12." (FAR 4.601(d)(5)(6); FAR 12; FAR 52.212-4)

**Derivation note (from dictionary):** "Not required for Mods means the value pulls from the basic document." (That is literally the dictionary's phrasing — it tells you mods inherit the base's value rather than reporting their own.)

### v1.4 codes
| Code | FPDS short description |
|---|---|
| A | Commercial Item |
| B | Supplies or services pursuant to FAR 12.102(f) |
| C | Services pursuant to FAR 12.102(g) |
| D | Commercial Item Procedures not used |

Code `E` ("DoD Section 803 CSO Procedures") appears in the parquet (27 rows) but is **not** in v1.4 — a post-v1.4 addition.

(Description text for A and D was reworded in the USAspending download mid-FY2022 — filter on code, not description.)

---

## `multiple_or_single_award_idv_code` / `multiple_or_single_award_idv`  (element 6E)

**Definition (FPDS):** "Indicates whether the contract is one of many that resulted from a single solicitation, all of the contracts are for the same or similar items, and contracting officers are required to compare their requirements with the offerings under more than one contract or are required to acquire the requirement competitively among the awardees." (FAR 16.5)

**Requirement State (v1.4):** Only `BPA` and `IDC` bases are `R` (required). Everything else — PO, DO, BPA Call, FSS (derived to Multiple), GWAC (derived to Multiple), DCA — is `N`.

**Why it's 100% null in our parquet:** this field is stored on the IDV/BPA record itself. Our parquet is `award_or_idv_flag='AWARD'` only, so by definition it contains zero IDV records, and therefore zero rows where this field is populated. To get multi-vs-single-award IDV information, pull IDV-side records from the raw USAspending download or the FPDS ATOM feed.

Values when populated: `M` (Multiple Award) and `S` (Single Award).

---

## Minimal recipe for a "was this competed" feature

```python
import polars as pl

df = pl.scan_parquet("data/clean/procurement_data.parquet").with_columns(
    competed=pl.col("extent_competed_code").is_in(["A", "D", "F", "CDO"]),
    sole_source=pl.col("extent_competed_code").is_in(["B", "C", "G", "NDO"]),
    set_aside=(
        pl.col("type_of_set_aside_code").is_not_null()
        & (pl.col("type_of_set_aside_code") != "NONE")
    ),
    offers_received=pl.col("number_of_offers_received").cast(pl.Int64, strict=False),
)
```

Avoid using `offers_received` directly as a row-level feature unless you've filtered to `extent_competed_code in {'F','G'}` where coverage is > 90%.
