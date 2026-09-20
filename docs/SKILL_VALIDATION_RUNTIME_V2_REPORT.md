# Runtime-v2 Skill validation report

Date: 2026-09-20  
Branch: `codex/thesis-experiment-upgrades-industry`  
Generation model: `deepseek-flash`, thinking enabled, reasoning `medium`

## Executive conclusion

The completed experiment does **not** support freezing any of the nine extracted
Skills for transfer testing. The shared control mean semantic F1 was 0.6730. Eight
candidates reduced the mean score; the only positive candidate,
`group-comparison-effect-size-discipline`, changed mean F1 by only +0.0074 with a
95% interval of [-0.1512, +0.1660] and passed only one of four cases. Consequently,
the pre-registered gate selected 0/9 Skills and the source-test/target-test stages
were not started.

This is not evidence that automatic Skill extraction is impossible. It is evidence
that the current validation treatment—forcing one specialized Skill into every
validation case and every declared stage—is not a viable transfer mechanism. It
conflates Skill quality with trigger relevance and makes an irrelevant specialized
Skill pay a prompt/cognitive cost on cases where it should remain inactive.

## Completed experimental chain

- Source-train: 24 successful runs (8 cases × 3 repetitions), total recorded
  generation cost USD 1.042627.
- Extraction: 9 generalized candidate Skills.
- Leakage audit: passed with zero findings after the false-positive vocabulary
  rule was corrected.
- Source-valid generation: 120/120 canonical runs successful (12 shared controls
  plus 108 treated runs), total recorded generation cost USD 5.554855.
- Formal scoring: 120/120 predictions scored with
  `local-deepseek-v41-thinking-v2`; 4,361 judge calls, 3,188,004 prompt tokens,
  and 4,168,202 completion/reasoning tokens were recorded.
- Gate outcome: 0/9 candidates passed. Per protocol, no Skill package was frozen
  and no source-test or target-test was run.

## Candidate-level results

All deltas below are treated minus the shared control on semantic primary F1.
Intervals and unadjusted one-sample p-values use the 12 run-level differences per
candidate and are descriptive only: the same three control runs are reused across
candidates, so these are not nine independent experiments.

- `budget-reserve-and-final-answer`: delta -0.0412; 95% interval
  [-0.2043, +0.1219]; p=0.590; 3/4 cases positive; cost ratio 1.069.
- `screen-derived-metrics-for-generation-artifacts`: delta -0.1588; interval
  [-0.3290, +0.0114]; p=0.065; 1/4 cases positive; cost ratio 1.189.
- `stress-test-trend-claims`: delta -0.2595; interval
  [-0.5041, -0.0148]; p=0.040 unadjusted; 1/4 cases positive; cost ratio 1.054.
  This is the only nominally significant result, and its direction is harmful.
- `audit-categorical-redundancy-and-concentration`: delta -0.0207; interval
  [-0.1386, +0.0972]; p=0.706; 2/4 cases positive; cost ratio 1.162.
- `verify-label-against-content`: delta -0.1120; interval
  [-0.2893, +0.0652]; p=0.192; 1/4 cases positive; cost ratio 1.118.
- `check-systematic-missingness-before-group-comparison`: delta -0.0544;
  interval [-0.1942, +0.0854]; p=0.410; 1/4 cases positive; cost ratio 1.184.
- `scope-guard-on-stated-window`: delta -0.1528; interval
  [-0.3638, +0.0582]; p=0.139; 1/4 cases positive; cost ratio 1.053.
- `group-comparison-effect-size-discipline`: delta +0.0074; interval
  [-0.1512, +0.1660]; p=0.921; 1/4 cases positive; cost ratio 1.179.
- `escalate-integrity-defects-to-headline`: delta -0.1381; interval
  [-0.3079, +0.0317]; p=0.101; 1/4 cases positive; cost ratio 1.250.

## Variance and interpretation

The shared-control means by validation case were 0.7529, 0.8264, 0.7718, and
0.3410 for flags 9–12 respectively. Flag 12 also had the largest run-to-run
standard deviation (0.2092) and contained a 0.1000 control outlier. Every candidate
appeared to improve flag 12. Because the same weak control triplet was reused, that
pattern is more consistent with case-specific variance/regression to the mean than
with nine different Skills all transferring successfully to the same case.

The result therefore supports three claims:

1. Single-run improvements are not reliable evidence of transfer.
2. Unconditional Skill injection usually adds cost and often distracts the agent.
3. A specialized Skill must be evaluated jointly with its trigger/selector; forcing
   it into irrelevant cases tests prompt pollution, not the intended Skill system.

## Reliability corrections made during the run

Forty failed attempts were preserved outside the canonical experiment tree. They
all exposed the old generic `no terminal response` message. Canonical runs were
then completed successfully without overwriting successful artifacts.

An initial scoring pass produced 120 zero scores because authentication failures
were silently converted into empty Monte Carlo ratings. Those scores are invalid
and excluded. The valid `v2` scorer pass used an explicitly verified credential,
produced non-zero scores, and recorded real usage. The scorer now fails closed on
authentication/transport failure, and the PI stream adapter now preserves the real
provider error message.

## Minimal next experiment

Do not loosen the gate or repeat the same 120-run design. Redesign validation into
two separable tests:

1. **Trigger test.** Make the full candidate library available but do not force any
   Skill. Record which Skill was activated, at what stage, and the trigger evidence.
   Evaluate activation precision and the no-Skill rate on irrelevant cases.
2. **Conditional efficacy test.** Pre-register eligible cases for each Skill using
   information available before generation (task wording, schema, missingness, time
   fields, categorical fields). Only estimate a Skill's treatment effect on eligible
   cases. Keep ineligible cases for a no-harm selector test rather than demanding a
   positive score from forced injection.
3. **Mechanism test.** Convert high-value Skills from prose reminders into executable
   actions or checks. For example, a trend Skill should schedule a robustness query;
   a missingness Skill should run a missingness-by-group diagnostic. Record whether
   the prescribed action executed and whether its result entered the final answer.
4. **End-to-end transfer.** After both selector quality and conditional efficacy pass,
   freeze the package and compare `pi-core` with `pi-auto-skills` on untouched
   source-test and InsightEval target-test cases, with three repetitions and
   cost-normalized reporting.

## Thesis implication

The current negative result is useful pilot evidence but is not yet the thesis's
main positive result. It identifies the central research problem more precisely:
automatic extraction alone is insufficient; transfer depends on context-sensitive
Skill activation and executable realization. A defensible thesis contribution can
therefore be framed as a provenance-aware pipeline for extracting, selecting,
executing, and statistically validating transferable analysis Skills, with the
runtime-v2 experiment reported as the ablation demonstrating why unconditional
prompt injection fails.
