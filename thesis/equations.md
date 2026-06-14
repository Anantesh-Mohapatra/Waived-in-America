# Equations pending — Gap A

Three displayed equations belong in the Research Design section. EQ1 and EQ2 are composed as bracketed placeholders in the .docx (per user direction: avoid Word ↔ Google Docs sync friction during in-progress drafting). EQ3 (synthdid) has no placeholder in the .docx yet — the synth subsection is currently prose-only and a fresh tracked insertion is needed when the equation is added.

The plan: equations themselves go in as images. The breakdown paragraph below each entry is written as thesis prose for direct paste into the body, following the equation image.

## Placeholders in the .docx

| Slot ID | Placeholder string (verbatim) | Status in .docx |
|---|---|---|
| **EQ1** | `[insert equation EQ1: event study fixed-effects model]` | Present, centered, single paragraph |
| **EQ2** | `[insert equation EQ2: matched-controls per-NSN treatment effect]` | Present, centered, single paragraph |
| **EQ3** | *(no placeholder yet — insert into the Synthetic Controls subsection)* | Needs fresh tracked insertion |

## EQ1 — Event study fixed-effects model

**Location:** Research Design > Event Study subsection, immediately after the lead-in sentence ("The event study models each panel observation's outcome as a function of NSN fixed effects, fiscal-year fixed effects, and event-year dummies.").

**Existing prose to delete:** ¶106 and ¶107 (the two paragraphs immediately following the placeholder, beginning "The NSN fixed effect absorbs differences between NSNs…" and "Each event-year coefficient therefore measures…"). Their content is folded into the breakdown paragraph below.

**LaTeX:**

```latex
y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \beta_k \mathbb{1}\{e_{it} = k\} + \varepsilon_{it}
```

**Breakdown paragraph (paste after the equation image, replacing ¶106 and ¶107):**

In this specification, the outcome y is observed for each NSN i in each fiscal year t. The NSN fixed effect αᵢ absorbs differences between NSNs that do not change over the study period, like intrinsically higher price levels for certain products. The fiscal-year fixed effect γₜ absorbs shocks that hit all NSNs in a given year, such as a continuing resolution that delays procurement spending across the government. Each event-year indicator equals one when NSN i is at event year k in fiscal year t, and the coefficient βₖ measures the outcome at event year k relative to event year -1, which is omitted as the reference. The error term εᵢₜ captures variation in the outcome that the fixed effects and event-year dummies do not explain, and standard errors are clustered at the NSN level. The output is an event-time path: one coefficient per event year, with -1 normalized to zero.

## EQ2 — Matched-controls per-NSN treatment effect

**Location:** Research Design > Matched Controls subsection, immediately after the lead-in sentence ("For each treated NSN, the per-NSN treatment effect is the treated NSN's pre-to-post change minus the weighted average of the matched NSNs' pre-to-post changes:").

**Existing prose to delete:** ¶112 ("The pooled average treatment effect on the treated (ATT) is the average of the per-NSN effects across treated NSNs."). Its content is the final sentence of the breakdown paragraph below.

**LaTeX:**

```latex
\hat{\tau}_i = (\bar{y}^{\,\text{post}}_i - \bar{y}^{\,\text{pre}}_i) - \sum_j w_{ij}\,(\bar{y}^{\,\text{post}}_{ij} - \bar{y}^{\,\text{pre}}_{ij})
```

**Breakdown paragraph (paste after the equation image, replacing ¶112):**

Here τᵢ is the per-NSN treatment effect for treated NSN i. The terms ȳᵢ-pre and ȳᵢ-post are the treated NSN's average outcomes before and after the waiver, and ȳᵢⱼ-pre and ȳᵢⱼ-post are the analogous pre- and post-waiver averages for matched untreated NSN j. The weights wᵢⱼ are produced by the matching step, which selects the three nearest neighbors by Mahalanobis distance over the four pre-period covariates described above. The pooled average treatment effect on the treated reported in the body is the simple mean of τᵢ across treated NSNs.

## EQ3 — Synthetic difference-in-differences estimator

**Location:** Research Design > Synthetic Controls subsection. The equation goes near the top of the subsection. The current ¶114 needs partial replacement: keep the first sentence (citation + framing) and the last sentence (donor eligibility rule), drop the middle sentences that the breakdown paragraph subsumes.

**Existing prose to handle:** ¶114 contains four substantive sentences after the opening citation. Of those:

- "For each treated NSN, it constructs a synthetic comparison from a weighted combination of untreated NSNs." → **delete** (subsumed)
- "The weights are chosen so that the synthetic's pre-waiver outcome path tracks the treated NSN's pre-waiver outcome path." → **delete** (subsumed)
- "The per-NSN treatment effect is the post-waiver gap between the treated NSN and its synthetic, taking into account any gap that already existed prior to the waiver." → **delete** (subsumed)
- "Eligible untreated NSNs are restricted to those that observe the outcome in every event year the treated NSN does, which prevents the synthetic from being constructed by extrapolating across years of missing data." → **keep** (donor eligibility rule, not in the breakdown)

**Suggested lead-in** (replaces the opening sentence of ¶114):

> Synthetic difference-in-differences (Arkhangelsky et al. 2021) extends the synthetic control method by combining unit weights with time weights inside a two-way fixed-effects regression. For each treated NSN, the estimator solves the following weighted least-squares problem:

**LaTeX:**

```latex
(\hat{\tau}, \hat{\mu}, \hat{\alpha}, \hat{\beta}) = \arg\min_{\tau, \mu, \alpha, \beta} \sum_{i=1}^{N} \sum_{t=1}^{T} \left( Y_{it} - \mu - \alpha_i - \beta_t - W_{it}\,\tau \right)^2 \hat{\omega}_i\, \hat{\lambda}_t
```

**Breakdown paragraph (paste after the equation image; the surviving donor-eligibility sentence from ¶114 then follows):**

In this expression, Yᵢₜ is the outcome for NSN i in fiscal year t, and Wᵢₜ equals one when NSN i is the treated NSN and t falls in the post-waiver window. The terms αᵢ and βₜ are NSN and fiscal-year fixed effects, and μ is a common intercept. The unit weights ωᵢ are zero for the treated NSN and non-negative across the donor pool, chosen so that the weighted average of donor pre-waiver trajectories tracks the treated NSN's pre-waiver trajectory. The time weights λₜ are zero in post-waiver periods and chosen so that the weighted average of pre-waiver outcomes approximates each donor's post-waiver mean. The estimated treatment effect τ is therefore the post-waiver gap between the treated NSN and its synthetic comparison, net of any pre-waiver level difference absorbed by the time weights.

## Workflow for the future session

1. Open `Thesis_Writing/Honors Thesis - First Draft.docx` in Word.
2. For EQ1 and EQ2: find each placeholder string via Word's Find (Ctrl+F). Use Insert → Equation → "Insert New Equation" (OMML) and either type the equation directly or paste the LaTeX (recent Word versions accept LaTeX-style entry; alternatively, paste through MathType). Delete the placeholder paragraph.
3. For EQ3: open the Synthetic Controls subsection. Replace the first sentence at ¶114 with the suggested lead-in above, insert a new centered paragraph for the equation, and paste the breakdown paragraph immediately after it. The remainder of the existing ¶114 prose follows unchanged.
4. After each equation, paste the corresponding breakdown paragraph as the next paragraph in the body. The breakdowns are written to read as continuous thesis prose, so no further editing should be needed.

## See also

- `Thesis_Writing/AUDIT_TRACKER_2026-05-24.md` — Major item #1 tracks this task.
- `Thesis_Writing/_audit_unpacked/thesis_plain.txt` — paragraph-numbered extract used to locate ¶105, ¶111, ¶114.
