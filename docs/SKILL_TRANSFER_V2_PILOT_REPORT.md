# PI Skill transfer v2 pilot report

Date: 2026-09-21

## Scope

This pilot tests the first automatically generated physical PI Skill on one held-out source-validation case (`flag-9`). It is a pipeline and mechanism check, not evidence of general transfer.

- Creator input: 24 scored `source-train` episodes.
- Creator output: two physical Skills, `derived-metric-validity-audit` and `composition-artifact-check`.
- Pilot candidate: the first manifest entry, `derived-metric-validity-audit`.
- Runtime comparison: PI autonomous research loop without Skills versus the same loop with one Skill directory available.
- Replication: three independent generation runs per arm.
- Formal metric: one frozen local DeepSeek v4.1-thinking judge run per prediction, `semantic.primary.f1` only.
- Model: `deepseek-flash`, reasoning `medium`, thinking enabled.

## Integrity checks

The Skill Creator inspected ten scored training episodes for the accepted Skill set. The accepted directory passed the training-literal audit. Both generated executable Skills passed their own offline Python test assets. Validation plans isolate one Skill and create exactly three control and three treated runs. All six generations and all six scoring jobs completed successfully.

The selected Skill is optional runtime knowledge. The base agent initially receives only Skill metadata. It must call `read_skill` before seeing the full instructions and must read the Skill before it can call `run_skill_python`. This pilot does not inject the Skill body into the system prompt.

## Results

| Arm | Run | Recall | Precision | F1 | Generation cost | Calls | Elapsed | Read | Executed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | 1 | 0.7600 | 0.8000 | 0.7795 | $0.024689 | 28 | 106.9 s | 0 | 0 |
| Control | 2 | 0.9200 | 0.6833 | 0.7842 | $0.013342 | 16 | 60.8 s | 0 | 0 |
| Control | 3 | 0.8399 | 0.8000 | 0.8195 | $0.016867 | 15 | 67.7 s | 0 | 0 |
| Skill | 1 | 0.8800 | 0.7667 | 0.8194 | $0.040774 | 32 | 162.3 s | 1 | 1 |
| Skill | 2 | 0.8800 | 0.7167 | 0.7900 | $0.027698 | 19 | 100.4 s | 1 | 0 |
| Skill | 3 | 0.7800 | 0.7500 | 0.7647 | $0.028029 | 32 | 107.4 s | 1 | 1 |

Aggregate results:

- Control mean F1: 0.7944; Skill mean F1: 0.7914; delta: -0.0030.
- Recall changes from 0.8400 to 0.8467 (+0.0067); precision changes from 0.7611 to 0.7444 (-0.0167).
- Run-index deltas are +0.0399, +0.0058, and -0.0548. The sign is not stable.
- Mean generation cost rises from $0.01830 to $0.03217 (1.758x).
- Mean model calls rise from 19.7 to 27.7 (1.407x).
- Mean elapsed time rises from 78.5 s to 123.4 s (1.572x).
- Skill read rate is 3/3. Script execution rate is 2/3.
- Total validation generation usage is 142 calls and $0.151399. Formal scoring used 186 judge calls.

## Trajectory interpretation

The runtime mechanism works as intended. In all treated runs the agent independently opened the Skill after beginning its analysis; in two runs it then executed `scripts/metric_validity_audit.py`. The Skill changed the evidence collected: treated trajectories explicitly audited `closed_at - opened_at`, detected that `closed_at` duplicates `sys_updated_on`, checked lattice structure, generator fit, subgroup effects, and censoring, then downgraded resolution-time claims.

That behavior is analytically defensible, but it was not a stable benchmark gain on this case. `flag-9` mainly rewards identifying the July-August hardware-volume burst and reproducing the reference TTR observation. The Skill added a validity challenge to the TTR field and consumed one of six question slots. Across three runs this slightly increased recall but reduced precision and increased cost. One execution helped, one hurt, and the read-only run was nearly neutral, so neither execution nor mere reading has a consistent effect in this pilot.

The correct conclusion is therefore not that the Skill is ineffective. The supported conclusion is narrower: the physical Skill mechanism activates and changes analysis, but this candidate has no demonstrated net score benefit on the single tested case and is too expensive in its current broad-trigger form.

## Decision

Do not freeze or promote `derived-metric-validity-audit` from this pilot. Preserve it as a candidate. Do not test more candidates until the following minimum upgrades are made:

1. **Narrow the trigger.** The current description says to use the Skill before interpreting any derived metric. Require stronger observable evidence, such as duplicate/affine timestamp fields, impossible values, a near-perfect fit, or subgroup claims whose metric validity is genuinely load-bearing.
2. **Add a relevance/exit gate.** After a cheap precheck, the agent should stop the Skill path when no artifact signature is found instead of spending a full research question.
3. **Cap Skill overhead.** Reuse existing query results, allow at most one dedicated audit question, and execute the script only when its required columns and suspected derived metric are present.
4. **Separate two hypotheses in the next experiment.** Measure selection quality first (whether the agent reads the right Skill), then conditional efficacy on pre-registered eligible cases. Do not average effects over cases where the Skill has no real opportunity to help.
5. **Run the smallest credible transfer test.** After these changes, use at least four unseen validation cases with three runs per arm. Freeze the Skill only if the case-level mean effect is positive on at least two cases, mean delta is positive, and the cost ratio meets the pre-registered bound.

The untested `composition-artifact-check` remains a separate candidate. It should not inherit the result of this pilot, but it also should not be run until the relevance and cost controls above are implemented.

