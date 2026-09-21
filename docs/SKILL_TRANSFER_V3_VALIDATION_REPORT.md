# PI Skill transfer v3: source-valid report

Date: 2026-09-21

Repository commit used for generation: `f772ab3ba87ad6adf3ae1c9cec3cf5f23bf6a0dc`

Generation model: `deepseek-flash`, thinking enabled, reasoning `medium`

Scorer: `local-deepseek-v41-thinking-v2`, primary semantic F1 only

## Decision

None of the three automatically generated candidate Skills passed the frozen source-valid gate. No Skill set was frozen, and the untouched source-test cases (`flag-13` onward) were not opened.

This is a valid negative selection result, not a failed runtime. All 48 generation runs and all 48 primary scoring runs completed. The optional Skill mechanism worked: the Agent chose whether to read and execute each Skill, and all 14 executable-Skill attempts succeeded. The failure is empirical: making these Skills available did not produce a stable, case-general improvement.

## Design

- Validation cases: `flag-9` through `flag-12` (the complete frozen `source-valid` split).
- Arms: shared no-Skill PI core control, plus one treated arm for each candidate Skill.
- Repetitions: three independent Agent runs per case and arm.
- Search cap: up to 25 research questions (`5 × 5` proposal budget), with autonomous early stopping.
- Total generation runs: 12 control + 36 treated = 48.
- Candidate gate: every case must have three runs per arm, positive F1 delta, and cost ratio at most 1.5; overall mean delta must be non-negative.
- The gate was fixed before scoring. It was not relaxed after seeing results.

## Aggregate results

| System | Mean F1 | SD across runs | Precision | Recall | Mean generation cost | Mean questions | Mean calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| PI core (no Skill) | 0.7252 | 0.1371 | 0.6777 | 0.7974 | $0.0340 | 9.1 | 36.7 |
| + composition-adjusted-segment-contrast | 0.7267 | 0.1240 | 0.7125 | 0.7650 | $0.0469 | 9.8 | 45.9 |
| + derived-column-integrity-audit | 0.6662 | 0.1423 | 0.6502 | 0.6926 | $0.0268 | 6.8 | 35.8 |
| + trend-claim-robustness | 0.7115 | 0.1731 | 0.6694 | 0.7737 | $0.0360 | 9.9 | 41.1 |

Mean ITT deltas versus the same shared control were:

- composition: **+0.0015**;
- derived-column audit: **-0.0590**;
- trend robustness: **-0.0137**.

The composition Skill was effectively neutral on average, but that average hides three negative cases and one positive case. The other two Skills were negative on average.

## Case-level results

| Skill | flag-9 | flag-10 | flag-11 | flag-12 | Mean | Gate result |
|---|---:|---:|---:|---:|---:|---|
| composition-adjusted-segment-contrast | -0.0004 | -0.0298 | -0.0093 | +0.0455 | +0.0015 | Fail |
| derived-column-integrity-audit | +0.0313 | -0.0732 | -0.1130 | -0.0809 | -0.0590 | Fail |
| trend-claim-robustness | +0.0013 | +0.0829 | -0.0740 | -0.0652 | -0.0137 | Fail |

The run-to-run SD within a case was often larger than the mean delta. For example, control SD was 0.0997 on `flag-10`; the composition treated SD there was 0.1734. With three repetitions, isolated gains cannot be treated as deterministic improvements.

## Statistical inference

The positive Composition point estimate is not statistically distinguishable from zero. Treating case as the analysis block gives:

- composition: mean delta `+0.0015`, 95% CI `[-0.0491, +0.0521]`, exact blocked permutation `p=0.9636`;
- derived-column audit: mean delta `-0.0590`, 95% CI `[-0.1586, +0.0406]`, `p=0.0503`;
- trend robustness: mean delta `-0.0137`, 95% CI `[-0.1294, +0.1019]`, `p=0.6523`.

The blocked interval first computes the three-run treated-minus-control mean within each of the four cases, then estimates uncertainty across those four case effects. The exact randomization test enumerates every 3-vs-3 assignment within each case and combines the four blocks (160,000 assignments per Skill).

Even Composition's only positive case (`flag-12`, delta `+0.0455`) has a 95% Welch interval of `[-0.1408, +0.2318]`. Therefore this round supplies neither evidence of a significant improvement nor proof that Skill availability is harmful. The negative Derived result is the strongest warning signal, but its case-blocked interval still crosses zero.

## Skill activation and runtime integrity

| Skill | Read rate | Run-level execution rate | Mean paired delta when read |
|---|---:|---:|---:|
| composition-adjusted-segment-contrast | 5/12 (41.7%) | 5/12 (41.7%) | +0.0742 |
| derived-column-integrity-audit | 10/12 (83.3%) | 4/12 (33.3%) | -0.0628 |
| trend-claim-robustness | 9/12 (75.0%) | 2/12 (16.7%) | +0.0034 |

There were 14 script calls across 11 treated runs, and all 14 succeeded. Thus the stable `run(sql_results, skill_args)` ABI fixed the earlier execution failure. Reads and executions were autonomous: no Skill body was injected into every run, and several runs correctly completed without reading or executing the available Skill.

Activation-conditioned deltas are diagnostic, not causal estimates. The Agent decides to activate a Skill after observing the case and its own trajectory, so activated runs are a selected subset. In particular, the positive mean among composition executions does not by itself prove that execution caused the gain.

## Question budget

The 25-question cap was not the limiting factor:

- 47 of 48 runs stopped early;
- mean completed questions were 9.1 for control, 9.8 for composition, 6.8 for derived-column audit, and 9.9 for trend robustness;
- only one trend run on `flag-11` reached 25 questions.

Increasing the cap again is therefore not the next useful intervention. The main problem is what the Agent chooses to investigate and retain, not an inability to ask enough questions.

## Why the candidates failed

### 1. The Creator found real analytical techniques, but not consistently transferable task improvements

All three Skills are substantive procedures with executable checks, not generic reminders. The problem is that a sophisticated diagnostic can still be a poor intervention for the current task. Availability changed the research path, but not reliably in the direction rewarded by correct task coverage.

### 2. The derived-column audit can hijack the whole answer

On `flag-10`, one executed run recovered a strong mechanical rule for TTR and correctly challenged a naive operational interpretation. However, it then concentrated the answer around metric invalidity. The benchmark task also required coverage of volume/TTR correlation, category-wide behavior, and agent productivity. The treated run covered some of these but omitted part of the requested factor set, reducing both recall and precision relative to a strong control run.

The same Skill was read in 10 of 12 runs and executed on `flag-12`, whose primary task is assignment imbalance rather than validating a duration metric. This is over-broad applicability. Its execution-conditioned mean delta was -0.1426 across four selected runs.

### 3. Trend and composition Skills show localized promise, not stable transfer

The trend Skill improved `flag-10` by +0.0829 but lost on `flag-11` and on the non-trend `flag-12`. It was read on every `flag-9` run even though that task concerns a hardware anomaly in a window. The Skill usually was not executed, but reading it still altered question selection.

The composition Skill improved the assignment-imbalance case (`flag-12`, +0.0455). Its five executed runs had a mean paired delta of +0.0742, including large gains on one `flag-10` and one `flag-11` run. It also produced one substantial loss, and its ITT result remained neutral. This makes it the strongest candidate for redesign, not a validated Skill.

### 4. Skill output is not sufficiently subordinated to the original goal

The current Skills contain early-exit and redirect logic. That is useful locally, but the Agent sometimes treats a local diagnostic verdict as the new global research objective. A Skill should revise or invalidate one claim while preserving coverage of the user's remaining questions. This is especially important for data agents: detecting that one metric is synthetic does not remove the obligation to report the descriptive trends and other requested factors that remain answerable.

### 5. The current gate is intentionally conservative, and stochasticity is material

The rule requiring a positive delta on every validation case prevented a near-zero average from being promoted. That was the correct decision for this round. At the same time, per-case cost and score estimates from only three runs are noisy: composition failed the `flag-9` cost gate even though it was never read there, showing that the observed 1.95× cost ratio is stochastic run variation rather than Skill execution overhead.

The gate should not be changed retroactively. Future protocols should model case and run variance explicitly rather than interpreting each three-run point estimate as deterministic.

### 6. The deeper mismatch is validator Skills versus discovery Skills

The three candidates mainly answer “is this claim robust or measurable?” The held-out tasks primarily require the Agent to discover where an anomaly occurs: the relevant time window, category, location, volume shift, text theme, and plausible driver chain. PI core already performs substantial metric skepticism. Adding another validator can therefore duplicate a capability the base Agent already has without increasing missing-insight recall.

This distinction is clearest in the logs. The Derived Skill can recover an impressive mechanical timestamp rule, but the trajectory then narrows around that rule and omits other requested dimensions. Composition is closest to a discovery Skill and shows localized gains on assignment/segment cases, but not a stable cross-case effect.

The earlier proposal to add more negative triggers is therefore only a possible efficiency refinement, not the main research upgrade. This experiment does not show that restrictions would improve performance.

## Minimum next upgrade

1. **Keep the Agent architecture and executable ABI unchanged.** Autonomous discovery, reading, execution, provenance, and error accounting worked.
2. **Change the Creator's target from validators to discovery operators.** Use source-train scorer matching matrices and high/low trajectory differences to identify repeatedly missed insights, then generate procedures that search time windows, segments, locations, volume shifts, and text themes and return ranked hypotheses with evidence.
3. **Keep the Agent free.** Discovery Skills should expose optional tools and compact candidate evidence, not a mandatory workflow or a ready-made global conclusion. Negative triggers may be retained as metadata, but they are not the main intervention.
4. **Run a causal content ablation.** Compare no Skill, the current validator Skill, a new discovery Skill, and an equal-length irrelevant/placebo Skill. The placebo arm tests whether mere Skill availability or reading creates attention/cost side effects.
5. **Use source-train cross-validation for the next Creator iteration.** Tune Skill content on folds of `flag-1`–`flag-8`; treat the already used `flag-9`–`flag-12` split as validation history, not new evidence.
6. **Pre-register the next selection rule before new API calls.** Retain Skill availability (ITT) as the primary effect. Add case-blocked confidence intervals and a blocked permutation test; keep read/execution-conditioned analysis secondary because activation is post-treatment.
7. **Do not open source-test yet.** Only a new candidate set that passes the pre-registered validation procedure should be frozen and evaluated on untouched `flag-13`–`flag-17` (or a separately frozen test subset).

## Reproducibility artifacts

- Candidate Skills: `results/skill-transfer-v3/created-skills/`
- Creator provenance: `results/skill-transfer-v3/created-skills/*/provenance.json`
- Validation plan: `results/skill-transfer-v3/validation_plan.json`
- Collected scores: `results/skill-transfer-v3/validation_records.json`
- Frozen decisions: `results/skill-transfer-v3/validations.json`
- Immutable run manifests, trajectories, predictions, usage, and score files: `results/experiments/skillval-native-v3-q25-*`

Generation consumed 1,913 model calls across 48 runs and cost $1.7236 according to the recorded DeepSeek usage. The scorer produced 48 immutable score files using 2,142 judge calls. No generation or scoring run failed.
