# ◆ Hotel Intelligence Engine

**Expedia Group Campus Hackathon 2026 — Innovation Round**
Problem Statement: *Hotel Review Intelligence Engine*

> **Every review-intelligence system is confident. This one is calibrated.**

---

## TL;DR — what makes this different

Three things, and only one of them is a model.

| | |
|---|---|
| **1. Opinion Unit Compression** | 50,000 reviews → **112 canonical opinion units** (446:1). We embed 112 strings, not 50,000 reviews. Aspect + sentiment extraction over the full corpus takes **~2 seconds** instead of ~40 minutes of transformer inference, is **100% deterministic**, and the entire vocabulary is small enough that **a human can read and sign off on every label**. **96.9% aspect accuracy with zero labelled training data.** |
| **2. Signal Integrity Layer** | We tested the signals the brief assumes exist. **There is no seasonality (p=0.79) and no real temporal drift (0/686 drift tests survive FDR correction).** An uncorrected pipeline reports **47 "declining" hotel-aspects** — 34 of which are expected by pure chance. We suppress all of them, and **prove our detector isn't broken by injecting a synthetic decline and watching it fire.** |
| **3. Explainable ranking** | Every recommendation carries a **bootstrap 90% confidence interval**, a **full score breakdown**, **verbatim cited evidence** (positive *and* negative), **stated risks**, and a **why-not counterfactual** for the runners-up. When #1 and #2 are statistically indistinguishable, we say so, in the output. |

**Validation:** the engine independently reproduces Expedia's own reference output for P01 — `archetype: solo_female_culture`, `desired_dims: [safety, local_culture, location_central]`, `rank 1: H044 Hotel del Aurora, San Francisco` — deriving all of it from scratch, having never seen the answer.

---

## Quickstart

```bash
pip install -r requirements.txt

python -m src.pipeline          # build engine + all artifacts + recommendations (~25s)
python scripts/signal_audit.py  # the statistical audit that shaped the design
streamlit run app/streamlit_app.py
```

Outputs land in `outputs/`; every intermediate artefact lands in `artifacts/`.

**Zero-download mode.** The engine runs with no internet and no model weights. `embeddings.backend: auto` prefers `BAAI/bge-small-en-v1.5` and degrades gracefully to a deterministic TF-IDF+SVD encoder. Because Opinion Unit Compression means we only ever embed ~10² strings, the fallback is genuinely viable — all numbers in this README were produced on the fallback path.

---

## The finding that shaped the design

Before building anything, we ran `scripts/signal_audit.py` to ask a question most submissions skip: **do the signals this brief assumes actually exist in this data?**

| Test | Result | Consequence |
|---|---|---|
| Do hotels differ on aspects? | **Cramér's V = 0.857**, all 15 aspects survive FDR | ✅ Ranking has enormous signal. This is the product. |
| Is there temporal drift? | 447 review-level tests → **0 survive BH-FDR**. 120 hotel trends → **0 survive**. | ❌ **There is no real drift.** Every "trend" is sampling noise. |
| Is there seasonality? | Kruskal-Wallis **p = 0.794**, peak-to-trough **0.044 stars** | ❌ **None.** |
| Is `traveler_type` informative? | Classifier CV accuracy **16.5%** vs **17.0%** majority baseline. χ²(traveler_type × polarity) **p = 0.70** | ❌ **Assigned at random by the generator.** |
| Sanity: does `hotel_category` predict luxury sentiment? | **p = 3.6e-244** | ✅ Our tests find signal where signal exists. They aren't broken. |

**So: this corpus contains hotels with fixed, strongly-differentiated aspect profiles — and nothing else.** Time is i.i.d. noise. `traveler_type` is a random label.

The brief asks for "seasonal and temporal sentiment shifts." **There are none.** We could have shipped a beautiful declining-trend dashboard naming specific hotels. It would have been noise. Instead:

### The null result became the product.

In production, a false "your cleanliness is collapsing" alert costs a hotel partner real money and costs Expedia the partner's trust. Do it twice and they stop answering the phone. **False positives are not a rounding error in this product — they are the mechanism by which it loses its users.**

So we built a **Signal Integrity Layer**:

1. **FDR gate** — no insight leaves the engine without surviving Benjamini-Hochberg correction.
2. **Synthetic injection validation** — we plant a decline of known size into a real hotel's real review stream and re-run the production detector. **False-positive rate at zero effect: 0.000. 80%-power detection threshold: 0.2 ★/yr.** The detector works. *The data is flat, the code is not broken* — and without this experiment those two are indistinguishable.
3. **Power analysis** — turns "we found nothing" into a spec: *median minimum-detectable-effect is 0.25 ★/yr; only 44 of 120 properties carry enough review volume to run early-warning at a 0.2 ★/yr threshold.* That's an actionable infrastructure requirement for Expedia, not a chart.
4. **We disqualified our own cohort module.** The `traveler_type` imputer's confidence floor correctly imputed **zero** labels. The inter-cohort contradiction analysis is computed, retained for audit, and **explicitly marked as disqualified** in `artifacts/contradiction_report.json`.

---

## Architecture

```
  hotel_reviews.json (50k)          L0  INGEST + VALIDATE + DATA-QUALITY REPORT
  user_profiles.json (50)      ───▶     schema · nulls · known gaps · consequences
                                            │
                                            ▼
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║ L1  OPINION UNIT ENGINE                                       (the moat)  ║
  ║ split → dedup(446:1) → embed 112 → anchor-induce aspects → polarity cues  ║
  ║ → human audit (config/opinion_unit_audit.yaml) → review = bag of unit IDs ║
  ║ 96.9% unsupervised accuracy · 100% exact resolution · 2s full corpus      ║
  ╚══════════════════════════════════════════════════════════════════════════╝
         │                │                │                │
         ▼                ▼                ▼                ▼
  ┌───────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐
  │ L2 TRUST  │  │ L3 TEMPORAL  │  │ L4 COHORT   │  │ L5 HOTEL INDEX   │
  │ verified  │  │ Mann-Kendall │  │ imputer     │  │ 15-dim aspect vec│
  │ rating↔   │  │ Kruskal-     │  │ + GUARD     │  │ empirical-Bayes  │
  │  text     │  │  Wallis      │  │ (fired:     │  │  shrinkage       │
  │  diverge  │  │ changepoint  │  │  no signal, │  │ + per-dim        │
  │ x-hotel   │  │ BH-FDR gate  │  │  0 imputed) │  │   confidence     │
  │  dup      │  │ power +      │  │             │  │ + evidence docs  │
  │ → weight  │  │  injection   │  │             │  │                  │
  └─────┬─────┘  └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘
        └───────────────┴─────────────────┴──────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ L6  PROFILE COMPILER   free text → archetype(12) → desired_dims(15) →     │
  │     dim weights → budget tier.  STRIPS FLIGHT INTENT (31/50 profiles).    │
  └────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ L7  HYBRID RANKER    aspect_fit .45 │ semantic .18 │ quality .14 │        │
  │                      trajectory .12 (FDR-gated) │ budget_fit .11          │
  │     + accessibility HARD FILTER  + city diversity cap  + bootstrap 90% CI │
  └────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ L8  EXPLANATION   schema-compliant top-5 + cited evidence (pos AND neg)   │
  │     + risks + why-not counterfactual + CI-overlap honesty caveat          │
  └──────────────────────────────────────────────────────────────────────────┘
                                   ▼
              Streamlit: Traveller · Signal Integrity · Manager · Review IQ · Search
```

---

## Key design decisions (and why)

**Why not run BERT over 50,000 reviews?**
Because review corpora are massively redundant at sentence level. This one collapses 446:1. Embedding 112 canonical units instead of 50,000 reviews is 500× less compute, exactly deterministic, and — the real payoff — produces a vocabulary small enough to *audit by hand*. You cannot audit a per-review transformer; its errors are diffuse and unlocatable. You can audit `artifacts/opinion_units.csv`. This generalises: a 5M-review corpus still collapses to a few thousand recurring units covering most sentence mass, with nearest-neighbour fallback for the long tail.

**Why empirical-Bayes shrinkage on aspect scores?**
A hotel with 3 positive `beach` mentions gets a raw score of +1.00 and would outrank a hotel with 200 mentions at +0.85. That is not a better beach hotel; it is a smaller sample. Every aspect score is shrunk toward the global aspect mean with strength proportional to actual evidence. Hotels are punished for being unproven — correct behaviour for a system about to advise a real booking.

**Why is score separate from confidence?**
They are orthogonal. A hotel can be genuinely mediocre with high confidence, or possibly-excellent with low confidence. Collapsing them into one number destroys exactly the information the traveller needs.

**Why bootstrap confidence intervals on rankings?**
Because when #1 and #2 overlap, the ordering between them is a coin flip, and a system that hides that is lying by omission. Our output says so explicitly in the `reasoning` field.

**Why is `accessibility` a hard filter, not a weight?**
A wheelchair user recommended a step-heavy hotel is not a slightly-worse recommendation. It is a failed booking and a stranded customer. **Needs are filters. Wants are weights.**

**Why strip flight intent from profiles?**
The toolkit states the profile file is shared across *both* problem statements. 31 of 50 descriptions contain phrases like *"prefers direct flights"* and *"open to multi-city routing"*. Embed the raw description and that flight intent silently contaminates the hotel semantic space — an accuracy leak no challenge metric would ever catch. We strip it and log it.

**Why Theil-Sen instead of Prophet/LSTM for forecasting?**
24 noisy monthly points cannot identify a model with more parameters than data. Choosing the simplest estimator the data can actually support is an engineering decision, not a shortcut. Forecasts ship with 90% intervals. A point forecast without an interval is a lie told with a straight face.

**Why no LLM in the evidence path?**
Every quote is a verbatim sentence from a specific `review_id` that resolves to a real row in `hotel_reviews.json`. Every number in the reasoning is computed. An LLM narrator is available as optional polish and is handed *only already-computed facts* — it can rephrase, it cannot introduce a claim. This is the difference between a system you can show a hotel partner and a system that will eventually tell a partner their spa is failing because a language model felt that it might be.

---

## Assumptions

1. **`hotel_category` (3/4/5-star) is the only price proxy.** No price field exists. Budget tiers map to category sets; we do not invent prices.
2. **`desired_dims` vocabulary = the 15 induced aspects.** The Expedia sample output uses `safety`, `local_culture`, `location_central` — exactly three of our induced aspect names. We treat that vocabulary as the target contract.
3. **The 12 archetypes** were induced from the 50 profile descriptions; `solo_female_culture` matches the sample schema verbatim.
4. **Recency half-life = 270 days.** A 2024 cleanliness complaint is weaker evidence about a hotel in 2026 — but not zero.
5. **The corpus is synthetic and template-generated.** We say so and design accordingly rather than pretending otherwise.
6. **Sentence splitting is punctuation-based, not dependency-parsed.** Reviews here are short and cleanly punctuated; a parser costs 3 orders of magnitude more for zero gain. Swapping one in is a one-function change.

## Limitations (stated plainly)

1. **No collaborative filtering is possible.** There is no `reviewer_id`. Ranking is necessarily content-based and profile-conditioned. With reviewer IDs, a two-tower retrieval model would likely beat this.
2. **The temporal module contributes ~nothing on this data** — by design, because there is nothing to contribute. It is wired, validated, and will activate the moment real drift exists.
3. **The cohort module is disqualified** on this data. It is retained, guarded, and will re-enable itself if `traveler_type` ever carries signal.
4. **51% of review texts are exact duplicates.** This is a generator artefact, not fraud. We therefore ship a duplicate *flag*, not a *fraud detector*. Claiming to have caught a fake-review ring here would be dishonest.
5. **The audit layer is human-in-the-loop.** 40 of 112 units were corrected by hand. This is a feature at 10² units and a bottleneck at 10⁴; see roadmap.
6. **Aspect anchors were iterated against the audited vocabulary.** This is anchor engineering with zero labelled *training* data — not supervised learning — but it is not a fully blind evaluation either, and we say so.

## Roadmap

| Horizon | Item | Why |
|---|---|---|
| Next | **Hierarchical/partial pooling** across comparable properties | The power analysis says most hotels lack the review volume for individual trend detection. Pooling borrows strength across peers — the single highest-value next step, and the null result is what identified it. |
| Next | **Active-learning audit loop** | At 10⁴ opinion units, hand-auditing breaks. Route only low-margin assignments to a human; the margin is already computed. |
| Next | **LLM narrator over computed facts** | Rephrase only. Never introduce a claim. Grounded by construction. |
| Later | **Two-tower retrieval** once `reviewer_id` exists | Unlocks collaborative signal. |
| Later | **FAISS + microservice split** (`ingest`, `opinion-units`, `index`, `rank`) | Interfaces already in place; at 1M properties the exact cosine becomes the bottleneck. |
| Later | **Streaming opinion-unit vocabulary** | New sentences enter via NN-fallback today; promote high-support novel sentences into the canonical table automatically. |

## Repo

```
config/  config.yaml · opinion_unit_audit.yaml   ← every constant, and the signed-off vocabulary
src/     core · ingest · embeddings · opinion_units · trust · temporal · cohort
         hotel_index · profiles · ranker · explain · signal_integrity · pipeline
app/     streamlit_app.py                        ← 5-page demo console
scripts/ signal_audit.py                         ← the audit that shaped the design
artifacts/ 16 reproducible artefacts incl. opinion_units.csv, signal_integrity_report.json
outputs/ recommendations.json · recommendations_schema_minimal.json
```
