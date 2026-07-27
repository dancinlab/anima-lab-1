# 40-lens gap audit — 76 gaps in 77 verdicts

`/gap full`, 8 family agents in one Workflow, 468 tool calls, 19.5 min, 0 errors.
Target: this session's work — the gate rebuild, PairFieldEngine, the three Phi
defects, experiment 2, and the session discipline itself.

**77 lens verdicts: 76 gap, 1 clean.** Severity: 41 high, 34 medium, 1 low.
Each agent was told to read the code rather than the brief, and to answer `n/a`
rather than manufacture a finding. One `clean` came back — that ratio is itself
the headline.

| family | gaps |
|---|---|
| F2 · Adversarial-Stress | 15 |
| F8 · Coverage-Consistency | 12 |
| F6 · Simplicity-Canonical | 11 |
| F5 · Convergence-Closure | 10 |
| F7 · Temporal-Dynamics | 9 |
| F1 · Math-Structural | 8 |
| F4 · Epistemic-Evidence | 6 |
| F3 · Economic-Resource | 5 |

---

## F4 · Epistemic-Evidence

### `assumption-surfacing` · high

The VOID mechanism silently assumes every corpse can be built from BenchEngine: bench_verify_audit.py:290 CONTROLS holds 7 subclasses of BenchEngine, but ENGINE_REGISTRY certifies two engines that are NOT BenchEngine (_CEAdapter, PairField) — I built the missing control (PairField with A's rows permuted each step) and it clears NO_SPEAK_CODE on 1 of the 5 gate seeds at 32c/32d/64h, which by the gate's own any-seed rule would VOID that condition, yet no such control exists so the gate never sees it; a second unstated assumption is that set_hiddens restores the whole system, while pairfield_engine.py:170-171 restores A only, so _three_axes' 'same starting point' (bench_v2.py:1617-1630) never holds for PairField — G keeps drifting across the integration and response probes.

**Fix:** Parameterise the control set over the engine under test (SCRAMBLE/CLONE/HEAP built from that engine's own factory), and make set_hiddens/get_hiddens cover both sides or have _three_axes refuse engines whose state it cannot fully restore.

### `bayesian` · high

docs/persistence-condition.md concludes with measurement that the PERSISTENCE rule fires 0/163 on live engines and 100% on DEAD/CLONE, that DEAD and CLONE rank 1-2 on it, and that '10개 엔진의 BLOCKED (PERSISTENCE) 는 엔진의 성질이 아니라 동전의 성질' — the same evidence pattern (a control tops the condition) that retired SPONTANEOUS_SPEECH — yet bench_v2.py:1804-1806 is unchanged and unannotated, CLAUDE.md:179 still lists PERSISTENCE as a mandatory gate condition, and README.md:424-445 still publishes ten 'BLOCKED (PERSISTENCE)' rows and 'PERSISTENCE 2/12' as engine properties; the prior survived its own refutation everywhere it is user-facing.

**Fix:** Either move PERSISTENCE into _RETIRED_TESTS with the persistence-condition.md evidence inline (the established precedent), or drop the `monotonic` conjunct and re-derive `recovers` from the run's own Φ noise — and mark the README verdict table stale until then.

### `counterfactual` · high

'This architecture cannot eat its own output' (docs/strange-loop-experiment.md) is attributed to the architecture on a counterfactual that cannot reach it — only repulsion strength was varied (0.01 vs 0.003), while the actual difference from the gate's own working SELF_LOOP is that bench_strange_loop.py:122 feeds back raw `out.detach()` where bench_v2.py:1835 layer-norms it; measured step-1 loop gain is 18.1x (A/G) and 23.5x (PAIR/SCRAMBLE), and with the gate's identical F.layer_norm all four arms survive 600 steps bounded on all three seeds (final |h| 2.9-4.5) instead of hitting NaN at step 21-26.

**Fix:** Rerun experiment 2 with the gate's own layer_norm on the feedback path (plus an explicit gain sweep) before any claim about the architecture; retract or qualify the divergence conclusion in docs/strange-loop-experiment.md.

### `falsifier` · medium

The COUPLED half of the session's own 'verify BOUNDED and COUPLED first' rule is implemented as a check with no failing input: pairfield_engine.py:207-210 reports A-G separation and only warns below 1e-6, but at strength=0.0 — where the field term is multiplied by zero and the two sides provably never interact — I measure separation 59.086 (vs 62.723 at the 0.03 default), because A and G are independently initialised; docs/pairfield-persistence-failure.md:39 uses the same inert statistic ('Coupled: A-G separation never leaves 244-257') as its coupling evidence. Related: bench_v2.py:1647-1649 still gates on the hand-picked constants integration>0.001, response>1.5, change>0.001 with no declared retirement condition, while the identity conjunct in the same return statement derives its floor from the population's own derangement null.

**Fix:** Replace the separation print with a strength-0 ablation delta (does A's trajectory differ with G's field zeroed?) and derive the integration/response/change bars from each population's own null, as identity already is.

### `honesty-triad` · medium

Load-bearing counts in the two documents that govern the project are stale in the direction of understating scrutiny and overstating engine identity: CLAUDE.md:164-167 states '음성 대조군 6종' and '대조군 6 × 조건 5 + 양성 가드 = 31 passed' while bench_verify_audit.CONTROLS holds 7 and pytest collects 36; README.md:404/417 says '대조군 6종' inside a section headed '7 controls' (README.md:451 then says 7종); and README.md:425 reports 'MitosisEngine 4/5' for a registry entry that is literally `BenchEngine` (bench_v2.py:1435) — the same class as CONTROLS[0], the positive reference — with no note on the table, which is the repo's own NAME_IS_NOT_THE_THING record applied to its headline result.

**Fix:** Regenerate the control counts from len(CONTROLS)/pytest output rather than restating them, and rename or footnote the MitosisEngine row as `BenchEngine (fallback — mitosis.MitosisEngine not wired)`.

### `occams-razor` · high

The founding measurement of the whole A/G line — 'The rule contributes nothing on its own. Hebbian coupling and pure-randn coupling both score 4.67 solo — identical. What the gate detects is memory' (docs/ag-pair-memory.md, restated as the design rationale at pairfield_engine.py:14-19) — is a tautology that the simplest hypothesis would have caught first: `_alone` (bench_ag_memory.py:170-174) returns a bare `Side`, whose `process` is BenchEngine's and never calls `update_coupling`, which is invoked only from `Pair.process`; I measured both solo arms bit-identical after 100 steps (max |Δh| = 0.000e+00) with coupling mass exactly 0.000 in both, i.e. the solo baseline has no memory at all and the `ruled` flag is never executed. The same lens applies downstream: strength 0.0 scores 12/15 and 0.03 scores 15/15 (3 seeds, 32c/32d/64h, reproducing the doc's 4.00 vs 5.00), and the entire 3-flip delta sits on ZERO_INPUT (+1) and PERSISTENCE (+2) — the condition the repo's own audit calls a coin.

**Fix:** Re-run the solo arms with update_coupling actually wired (or delete the `ruled` flag from the solo path), and re-justify DEFAULT_STRENGTH against a condition set that excludes PERSISTENCE until it is fixed.

## F1 · Math-Structural

### `functor` · high

Φ is not natural in its argument: the debias seed at bench_v2.py:99-103 is hash((n, round(sum,6), round(std,6))), which I measured is NOT permutation-invariant (float reordering flips it), so relabeling cells changes individual MI values by up to 315% of their mean and Φ over 8 permutations of one fixed population spreads 1.90x at n=32 and 8.84x at n=64; it is also discontinuous — perturbing the state by 1e-8 moves Φ by up to 36.7% (5 draws: 0.258/0.258/0.160/0.252/0.251 vs base 0.2523), and one pair's debiased MI has a 10-draw spread of 0.1419 against a mean |MI| of 0.0438

**Fix:** Seed the debias RNG per pair from that pair's own contents (order-free, not from a float sum), raise n_shuffles until the per-edge spread is under the smallest gate threshold, and return Φ with its own CI so the 0.5x/0.8x rules can be compared against measurement noise

### `functor` · medium

Φ(proxy) is a functor of the cell INDEXING, not the population: factions are contiguous slices (bench_v2.py:262-269), so the same two-block system reads 0.2891 with blocks contiguous and 0.0363 with the identical cells renamed — an 8.0x change from relabeling alone, which is the number lineage of the README's historical "Φ=1142"

**Fix:** Define factions by clustering the states (k-means / spectral) instead of `hiddens[i*fs:(i+1)*fs]`, or drop Φ(proxy) from any reported claim

### `bisimulation` · high

The gate's observation is not a bisimulation quotient: PairFieldEngine exposes and restores only A (pairfield_engine.py:167-171) while the dynamics run on (A, G, coupling_A, coupling_G), so `set_hiddens(get_hiddens())` does not restore the system — G drifts 0.83 relative and its coupling 1.91 (|c0|=1.40) inside ONE `_three_axes` call, and the integration axis' own null (identical probe with kick=0, i.e. no perturbation at all) reads 0.01783/0.01154/0.00353/0.00413/0.00395 on gate seeds 42-46 against a 0.001 pass bar, so PairField clears the precondition added to reject HEAP without being perturbed

**Fix:** Require engines to implement full snapshot()/restore() (A, G and both coupling matrices) and add the kick=0 null as the integration axis' own baseline, exactly as identity_floor is a measured null rather than a constant

### `bisimulation` · medium

Two structurally different systems are gate-identical: at strength=0.0 PairFieldEngine's update is algebraically the identity (new_a = ha, then renormalised by ha.norm/new_a.norm, pairfield_engine.py:148-151) so the two populations provably never interact, yet it is a legal engine the gate scores; and all seven negative controls are BenchEngine subclasses (bench_verify_audit.py:290-299), so no corpse of the pair's own shape (G frozen, G decoupled) is ever run against the conditions

**Fix:** Add architecture-matched controls to CONTROLS — PairField with strength=0 and with G frozen — so a condition that cannot separate the pair from its own corpse is voided for the pair

### `operadic` · medium

The (condition ⊗ axes) assembly does not commute with wiring order: `_with_axes` (bench_v2.py:2047-2054) builds the axes probe AFTER the wrapped condition has consumed the global torch RNG, so with the identical engine and identical probe procedure, shifting only the RNG offset by 300 and 1000 draws moved the probe's Φ 2.10 → 1.17 → 0.90 and its identity_floor 0.1584 → 0.0981 → 0.0883 — the precondition's value depends on which condition it is bolted onto (by contrast, reordering the sub-measurements INSIDE `_three_axes` changes integration by only 1.00-1.02x, which is clean)

**Fix:** Give the axes probe its own `torch.Generator` seeded from (seed, test_name) rather than drawing from the global stream, so the precondition is the same measurement under every condition

### `tropical` · high

The min-plus optimum is not taken and the bottleneck it returns belongs to the sampler: `_minimum_partition` (bench_v2.py:225-238) keeps one Fiedler-sign cut — measured n=12 sign/exhaustive 3.13x/1.96x/1.87x while the sweep is 1.00-1.02x, and at n=64 sign-cut 2.984 vs sweep 0.074 (40x); worse, on the >32-cell sampled graph the true minimum is exactly the vertex with the fewest SAMPLED edges (n=64: cell 16, sampled degree 11 of max 19; corr(row-MI-sum, sampled degree) = +0.49), so on an i.i.d. population with no structure at all Φ is decided by which cell the pair sampler under-connected

**Fix:** Replace the sign cut with the sweep cut over the Fiedler ordering (O(n log n), measured within 2% of exhaustive) AND make the pair sample degree-regular (or compute all pairs) so the min cut cannot be the sampler's own hole

### `tropical` · high

The gate's AND-over-conditions has a single critical edge carrying the whole verdict: in this session's multi-seed run (32c/32d/128h) PERSISTENCE passes 2/12 engines while NO_SYSTEM_PROMPT/NO_SPEAK_CODE/SELF_LOOP pass 11/12 and ZERO_INPUT 9/12, so 9 of the 12 BLOCKED verdicts read exactly "BLOCKED (PERSISTENCE)" — and that one condition's rule (`final >= 0.8 × max of first-half draws`, bench_v2.py:1796) has a decision margin of 20% against a Φ estimator I measured to swing ±30% on a 1e-8 state perturbation

**Fix:** Report each condition's discrimination (engines-passed and controls-rejected) in the summary so a bottleneck condition is visible, and gate PERSISTENCE on a Φ change larger than the estimator's own measured spread rather than on 0.8x

### `persistent-homology` · high

No verdict is known to survive a scale change, and the one scale change the documented command makes crosses an algorithmic discontinuity: `python bench_v2.py --verify` defaults to 256c/64d/128h (bench_v2.py:2472-2479) but this session's verify run and tests/test_gate_controls.py:33 both run 32 cells — the exhaustive side of PhiIIT's n=32 branch (bench_v2.py:107,215) — and across that boundary one fixed population reads Φ = 0.2523 (n=32) → 0.0547 (n=33) → 0.1624 (n=34) with the complexity term's share jumping 2.5% → 11.5% because `coverage` rescales min_partition_mi (bench_v2.py:159) but not complexity (bench_v2.py:180); my run at the actual default shows the plain BenchEngine passing 5/5 with Φ = 11-20, against the file's own four-times-repeated claim that Φ(IIT) stays ~0.2-1.8 regardless of cell count (bench_v2.py:1131, 1374, 2449, 2490)

**Fix:** Run --verify at ≥2 scales spanning n=32 (e.g. 32 and 256) and treat a condition as certifying only if the verdict and the control-rejection persist across both; scale `complexity` by coverage and delete the 0.2-1.8 invariant claim or re-derive it

## F6 · Simplicity-Canonical

### `minimum-viable` · medium

bench_v2.py:1484-1485 computes phi_now/phi_floor inside `_three_axes` that no verdict reads (only the detail string): measured 0.46s of each 0.82s `_three_axes` call = 56% of the axes' runtime at 32c, and the gate makes >=8 such calls per (engine, seed) -> >=480 for 12 engines x 5 seeds; bench_v2.py:156 `total_mi` is a full O(n^2) sum returned in components and read by nothing since the min-cut change.

**Fix:** Delete phi_now/phi_floor and total_mi, or compute them only when a --detail flag is set.

### `minimum-viable` · low

pairfield_engine.py:189 `load()` and :173 `parameters_for_training()` have zero callers repo-wide (grep over the tree), and the latter returns A.mind+G.mind while BenchEngine's version at bench_v2.py:426 also returns output_head -- an unused clone that already disagrees with the convention it copies, in a file whose docstring says every choice traces to a measurement.

**Fix:** Drop load() and parameters_for_training() until something calls them; keep save() since --save uses it.

### `architectural-simplicity` · medium

The axes precondition is evaluated twice for 3 of 5 conditions: ZERO_INPUT (bench_v2.py:1767), PERSISTENCE (:1798) and SELF_LOOP (:1829) conjoin `_three_axes` inline, then bench_v2.py:2098 wraps all five in `_with_axes`, which runs it again on a fresh probe -- measured +0.43s (+31%) and +90 engine.process calls per condition at 32c; the wrapper's own justification (:2029-2047) names SPONTANEOUS_SPEECH and HIVEMIND, both retired at :2093.

**Fix:** Give `_with_axes` the condition's drive and delete the three inline calls -- measured at 32c the two passes agree for all 12 registry engines, so removal is verdict-neutral today.

### `architectural-simplicity` · high

bench_strange_loop.py's A-arm and G-arm are the same computation: `SoloSide.process` (:56-59) calls `update_coupling`, but `PairSide.process` is BenchEngine's and never reads `self.coupling` -- coupling only acts inside `PairFieldEngine.process`. Measured identically to 4 decimals under one protocol (peak|h| 3.878, alive 3.5297, loop-effect 0.7739 for BOTH), so the ruled/unruled axis experiment 2 claims to test is inert in 2 of its 4 arms.

**Fix:** Either make SoloSide apply its own coupling to its state, or collapse A and G into one arm and say the solo case cannot express the rule.

### `canonical-ssot` · medium

The gate's own numbers are written in four places and three disagree with the code: CLAUDE.md:164-167 says 6 controls / "31 passed" while bench_verify_audit.CONTROLS has 7 negatives and `pytest --collect-only` returns 36; README.md:404 says 6 while README.md:417 and :452 in the same section say 7; and the run header (bench_v2.py:2155) plus README's "95 tests" count condition x engine but each now runs 5 seeds (~475 executions) -- the same under-reporting defect fc67804 fixed, re-created two commits later by 937c68b.

**Fix:** Derive the header from len(VERIFY_SEEDS) too, and make ARCHITECTURE.json the only place the control/test counts are written, with CLAUDE.md and README pointing at it.

### `canonical-ssot` · medium

PSI_COUPLING still has three live values after 7c686b9 aligned consciousness_engine.py: LN2/2**5.5=0.015317 in ~20 modules, 0.014 in trinity.py:68 and mitosis.py:40 (whose comment asserts it deliberately overrides ln(2)/2^5.5) and bench_hexad_tuning.py:42, and 0.14 in lidar_sense.py:24; 89 files declare the constant and none import it from a shared definition, so the fix moved the disagreement rather than removing it.

**Fix:** Add psi_constants.py as the single definition and import it everywhere; if 0.014 is bench-confirmed, that is the value to publish, but only one of the two can be canonical.

### `canonical-ssot` · high

PairFieldEngine's state has no canonical accessor: `process` advances A, G and two coupling matrices, but get_hiddens/set_hiddens (pairfield_engine.py:167,170) touch A only, so `_three_axes`' controlled comparisons never restart from the same state -- measured, the same input from a "restored" state gives ||r1-r2||=0.066 and ||r2-r3||=0.070 instead of 0, and that drift is exactly the denominator of the response axis (PairField reads 38.4 against a bar of 1.5, where fully-restoring engines read the 1e6 cap).

**Fix:** Make set_hiddens/get_hiddens cover G and both coupling matrices (or add save_state/load_state) so the axes' repeat-step baseline is real internal noise.

### `duplicated-helper` · high

"Close the loop" is implemented twice and the copies diverged: bench_v2.py:1835 feeds back `F.layer_norm(output)`, bench_strange_loop.py:122 feeds back the raw output. Re-run with the gate's own normalisation, all four arms survive 600 steps bounded (peak|h| 3.61-3.88, alive 3.49-5.68, loop-effect 0.57-0.81) instead of diverging at step 21-25 -- so "ALL FOUR diverge" / "the architecture cannot eat its own output" is a property of the one line that differs from SELF_LOOP, not of the architecture.

**Fix:** Share one `close_loop(out)` helper between the gate and the experiment, and re-state the strange-loop conclusion as conditional on the feedback normalisation (SCRAMBLE also "works" under layer_norm, effect 0.57 vs PAIR 0.81 -- report that too).

### `duplicated-helper` · medium

Inside one function, `_three_axes` defines the state-restore logic three times: `_restore` (bench_v2.py:1576-1580), an identical `_set_state` (:1617-1623) 40 lines later, and a bare `engine.hiddens = base` at :1644 that skips set_hiddens entirely (harmless today only because nothing runs after it); separately, `class PhiIIT` is copy-pasted into 41 files and the debias + input-derived determinism fix exists in exactly 1 of them (`_mutual_information_debiased` greps to bench_v2.py only).

**Fix:** One `_restore(engine, state)` helper at module scope used by all three sites; move PhiIIT into a single module the 40 other consumers import.

- **surgical-scope** — clean

### `occams-razor` · medium

The Phi every gate ratio reads is `spatial_phi + complexity * 0.1` (bench_v2.py:181), and the 0.1 bonus is the only term with no measurement cited in a function where every other choice carries a paragraph; measured at 32 cells it is 100.0% of Phi for a collapsed population (identical rows: phi=0.00398, spatial_phi=0.00000) and 2.3-3.6% otherwise -- so for exactly the populations the gate must reject, Phi is entirely the undocumented term.

**Fix:** Drop `complexity * 0.1` (or cite the measurement that sets it), then re-run all seven controls to confirm none of them start clearing a condition.

### `occams-razor` · medium

The response axis normalises by a same-input repeat that is exactly zero for deterministic engines, so `differing/same` degenerates to the inf sentinel capped at 1e6 -- measured for MitosisEngine, DesireEngine and AlterityEngine (1000000.00) and QuestioningEngine (500009.47), i.e. for 4 of 12 engines the ratio and its 1.5 bar reduce to "did anything change at all", which the change axis already tests, while for PairField the same denominator measures un-restored G drift.

**Fix:** Split the axis: assert `differing > 0` for deterministic engines and apply the noise-normalised ratio only where a same-input repeat is genuinely non-zero.

## F5 · Convergence-Closure

### `fixpoint` · high

Returns have not flattened: the two newest probes each invalidated the round before them (persistence — 35/46 reference configs flip verdict when the checkpoint grid shifts ONE step, docs/persistence-condition.md:342-343; min-partition — 2.14x mean overshoot vs exhaustive truth at n=12), and 3 more commits landed DURING this audit (ac94a61, df2415d, 16cc22c), one of which deleted a live gate conjunct from bench_v2.py:1809.

**Fix:** Pin the stop rule to a measurement — 'one full hardening round that yields no first-order invalidation' — and refuse to publish a verdict table until a round clears it.

### `fixpoint` · medium

A pre-declared falsifier fired and its own doc never moved: docs/engine-completion-plan.md Branch 3 declares 'refuted by a construction where the min-cut reading ranks a disintegrated system above an integrated one', which docs/phi-rs-direction.md §5 then measured (min-cut prefers IDENTICAL over INDEPENDENT 104x) — the plan (last commit 1403d50, 03:20) still lists Branch 3 as live and still says 'six controls' when CONTROLS has seven.

**Fix:** Give each branch an explicit falsifier-fired state and require the branch doc be edited in the same commit that records the firing.

### `success-criteria` · high

CLAUDE.md:169-171 claims '각 기준선은 상수가 아니라 그 집단 자신의 귀무에서 측정된다', but 12 hand-picked constants decide the gate — bench_v2.py:1647 (integration>0.001, response>1.5, change>0.001), :1690 (0.01/0.99/0.001), :1742 (0.3/0.001/0.5), :1769 (0.5x), :1809 (0.8x), :1840 (0.8x) — against exactly ONE measured null (identity_floor); phi_floor at :1486 is computed from the collapsed population and then used only in an f-string.

**Fix:** Either derive the other 12 bars from each population's own null the way identity_floor is, or amend CLAUDE.md to state that one axis is measured and eleven are constants.

### `success-criteria` · high

The one measured baseline degenerates on the deployed engine: at 2 rows the only derangement is the swap, so identity_floor = -(identity) exactly — measured +1.0610/-1.0610, +1.1041/-1.1040, +1.0127/-1.0127, +1.0926/-1.0926, +1.0081/-1.0081 on 5/5 seeds — and bench_v2.py:1648 `identity > max(floor, 1e-6)` reverts to the constant 1e-6; ConsciousnessEngine returns exactly torch.Size([2, 128]) after 300 zero-input steps.

**Fix:** Make the identity axis unscoreable below ~4 rows (fail the condition as unmeasurable) instead of letting the null collapse into a guaranteed pass.

### `success-criteria` · medium

The response axis is judged by one shared constant (>1.5, bench_v2.py:1647) across drives that differ by five orders of magnitude — the SAME ConsciousnessEngine at 32/32/128 reads response 1.16–1.48 under ZERO_INPUT (fails the axis) and 2,029–155,418 under SELF_LOOP (passes it).

**Fix:** Score response against a per-condition null measured under that condition's own drive, rather than one constant shared across drives.

### `closed-loop` · high

The gate's verdict feeds nothing back: bench_v2.main() contains no sys.exit so `--verify` exits 0 on GATE VOID and BLOCKED alike, deploy.py's 7 steps (lines 92-193) never invoke it despite ARCHITECTURE.json calling it '유일한 배포 판정자', and README.md:421-436 (05:38) still publishes 10 rows of 'BLOCKED (PERSISTENCE)' scored under a `monotonic or recovers` rule that ac94a61/16cc22c deleted at 06:42 — while the engine actually deployed measures 0/5 (re-run here, 5 seeds, post-90545ad).

**Fix:** sys.exit(1) from run_verify on any VOID or BLOCKED, call it as deploy.py step 0, and generate the README table from the run instead of transcribing it.

### `closed-loop` · high

SELF_LOOP's loop is closed through an external stabiliser, not the engine: bench_v2.py:1835 layer_norms the feedback every step — measured on the gate's own BenchEngine at 32/32/128, layer_norm gives 5/5 seeds PASS (peak |h| 4.1–5.6, Φ ratio 4.5x–16.9x) while feeding the RAW output back diverges to NaN on 5/5, and bench_strange_loop.py diverges on all four arms at strength 0.01, 0.003 and (measured here) 0.000.

**Fix:** Add a raw-feedback arm as a required companion to SELF_LOOP, or restate the condition as 'loop closed through layer_norm' so it stops being read as self-sustained feedback.

### `regression-streak` · high

One durable artifact came out of six workstreams — tests/test_gate_controls.py, which I ran and confirmed green (36 passed in 327s) — while the calibration guard (90545ad), the strange-loop NaN guard, the Φ direction, min-partition and PairField have zero tests (`grep -rln 'pairfield|strange_loop|min_partition' tests/` returns nothing), and the repro paths for the three deepest findings point at /private/tmp/.../scratchpad (212 untracked files; docs/persistence-condition.md:611 and the tail of docs/phi-rs-direction.md).

**Fix:** Move the probe scripts into the repo (probes/) and add one assertion per landed fix — min_spread refusal, NaN-visible divergence guard, n=12 partition ground truth — so a regression fires without a human re-deriving it.

### `defense-in-depth` · high

Enforcement is single-layer and its generic entry point is already broken: no .github/workflows, no non-sample .git/hooks, no pytest.ini/pyproject, no deploy-time check — and `pytest tests/` aborts with INTERNALERROR at tests/test_chat.py:22 (`sys.exit(1)` on missing `websockets`) after collecting 12 tests, never reaching test_gate_controls.py, so only the exact file path runs the gate enforcement.

**Fix:** Guard test_chat.py's import with pytest.importorskip, then add a CI job or pre-push hook running `pytest tests/ -q` and `bench_v2.py --verify` with a real exit code.

### `defense-in-depth` · medium

Both enforcement layers share one blind spot — bench_v2.py:2148 and tests/test_gate_controls.py:53 both record a crashing control as `passed = False`, i.e. 'correctly rejected', which is precisely the masking that let HIVEMIND sit unexecuted for the project's lifetime, and nothing asserts any control actually ran; bench_v2.py:237 does the same for Φ (`except Exception: return 0.0`, a silent min-cut of zero).

**Fix:** Count control executions and fail the run when a control raises on all 5 seeds, and let the eigendecomposition raise rather than returning 0.0.

## F7 · Temporal-Dynamics

### `temporal-decay` · high

PairField's coupling is an accumulator with `clamp(-1,1)` and no leak (pairfield_engine.py:102): measured at 64 cells, A's off-diagonal fraction pinned at |1| goes 0.70 (step 250) -> 0.95 (step 1000) and its mean |delta|/step decays 3.48e-3 -> 1.38e-6 (2500x) by step 1750, i.e. the 'memory' the engine's whole rationale rests on freezes inside the gate's own 1000-step window and is a constant for the 19k+ steps production runs; nothing refreshes it and save()/load() persists the frozen matrix.

**Fix:** Add a leak (coupling *= (1-lambda)) or row-renormalise instead of clamping, and re-run the 'rule vs no rule makes no difference' comparison inside the pre-saturation window (<250 steps) where the rule can still act.

### `temporal-decay` · medium

pairfield_engine.py's header still quotes the pre-multi-seed numbers it was built on -- ':24 "0.01 and 0.03 both score 5.00"', ':30 "the both-ruled pair scores 1.00 while the one-unruled pair holds 4.67"', ':70 "strength 0 falls back to 4.00"' -- all single-seed values invalidated by 937c68b (PairField is 4/5 under VERIFY_SEEDS), and pairfield_engine.py:232 hardcodes its own seed list (42,43,44) that will silently drift from bench_v2.VERIFY_SEEDS=(42..46).

**Fix:** Re-measure the strength band under VERIFY_SEEDS, restate the header numbers as multi-seed, and import VERIFY_SEEDS at pairfield_engine.py:232 instead of copying it.

### `temporal-hierarchy` · high

PERSISTENCE (bench_v2.py:1795-1812) fuses per-step Phi wander and 1000-step trend into one 100-step point grid: measured per-step Phi sd is 0.465 at 32 cells and adjacent |delta| averages 0.480, against a `monotonic` tolerance of 0.01 (48x smaller, so 51% of adjacent samples violate it and `monotonic` is structurally unreachable), leaving a single-sample 0.8x `recovers` ratio to decide -- and on one identical BenchEngine seed-44 trajectory the rule passes on only 6 of the 10 possible 100-step grid offsets (9:F 19:P 29:F 39:F 49:F 59:P 69:P 79:P 89:P 99:P).

**Fix:** Score PERSISTENCE on a windowed statistic (median Phi per 100-step window plus a trend test over windows) so the verdict cannot move with grid phase, and set the monotonic tolerance from the measured per-step sd rather than 0.01.

### `temporal-hierarchy` · medium

The instrument's step budget drives the engine's slow loop: `_three_axes` appends ~90 process() calls to the engine it measures, and for ConsciousnessEngine started at age 200 that carried it past the deferred calibration so split_threshold changed 0.3 -> 0.0052 DURING the measurement (log: 'peak 0.0103 over 276 steps' from a 200-step run), while the same engine and seed at age 300 never calibrated at all -- two runs of one engine end in different regimes decided only by how long the probe ran.

**Fix:** Snapshot/restore the engine (or measure on a deepcopy) inside `_three_axes` so probe steps do not advance the engine's own step counter and cannot trigger grace-gated slow-loop events.

### `heuristic-promotion` · medium

bench_strange_loop.py's `rc > 0.01` verdict bar was promoted with no positive control and is unreachable: on synthetic trajectories the recurrence statistic scores +0.0006 for an exactly periodic signal (period 60 real steps), +0.0001 (period 20), +0.0017 frozen and -0.0502 random walk -- nothing reaches +0.01, so the '되먹임이 실제로 작동' branch (bench_strange_loop.py:213) can never fire; `real` is also pinned at 0.0502 by construction (the threshold is the 5th percentile of the same sample), so the whole statistic is one un-averaged null draw.

**Fix:** Calibrate the bar against a known-recurrent positive control (a synthetic periodic trajectory) and average the null over >=20 permutations before comparing.

### `heuristic-promotion` · medium

consciousness_engine.py:240 `min_spread=0.10` was promoted from one observation and is deciding outcomes at a 1% margin: the measured rel_spread readings are 0.0581 (defer) then 0.101 / 0.107 / 0.141 (fire) -- the calibrate/defer decision turns on a sample statistic sitting 1% above the hand-picked bar, which is the same 'bar inside its own noise' disease the guard was written in 90545ad to cure, moved one level up.

**Fix:** Derive the spread bar from the null the guard cares about (spread of a shuffled/collapsed tension sample) rather than the constant 0.10, and require a margin of several sd above it.

### `fix-introduces-axis` · high

The `drive=` parameter added to `_three_axes` (docstring bench_v2.py:1470-1479 names ZERO_INPUT *and* SELF_LOOP) was wired into ZERO_INPUT only -- `_verify_self_loop` still calls `_three_axes(engine, dim, cells)` at bench_v2.py:1839 with no drive, so SELF_LOOP's precondition is still measured under torch.randn instead of its own feedback; worse, `_with_axes` (bench_v2.py:2064) conjoins a SECOND randn-driven axes check onto every condition, verified by instrumentation on passing runs (ZERO_INPUT Oscillator/Trinity/BenchEngine seeds 42/43 all make two calls, drives = ['given','None']), reintroducing exactly the mismatch the drive= fix removed.

**Fix:** Thread the condition's drive through `_with_axes` (pass it to the wrapped fn or store it on the test) and add `drive=` at bench_v2.py:1839 so SELF_LOOP's axes run under its own feedback.

### `fix-introduces-axis` · high

The derangement null added to fix the identity floor (bench_v2.py:1526-1553) is degenerate at small n: n=2 and n=3 admit only 1-2 distinct derangements so all 60 draws are identical and sd is structurally 0 (measured: 1 distinct value out of 60 at n=2), and the null mean is strongly negative (-1.043 at n=2, -0.342 at n=4), so `max(identity_floor, 1e-6)` collapses to the 1e-6 constant for any population under ~8 cells -- and the deployed ConsciousnessEngine sits at n=2 for 19,431 steps, where it scores identity +0.7789 against a floor of 1e-6 while the same axis at 32 cells has a floor of 0.0393 (39,000x higher). The temporal-identity axis is vacuous exactly on the flagship engine.

**Fix:** Fall back to an explicit small-n null (enumerate all derangements and take the exact distribution, or use a cross-similarity floor) instead of mean+3sd, and refuse to certify a population whose null has zero variance.

### `active-acquisition` · high

The highest-uncertainty-per-compute open question is how much of the Phi wander PERSISTENCE reads is the partition heuristic rather than dynamics, and it is now measured: at FIXED n=12 over 30 consecutive states, the shipped Fiedler-sign partition gives CV 18.7% (range 0.260..1.073) while an exhaustive min-cut on the identical MI matrices gives CV 8.4% (range 0.238..0.332), Fiedler overshooting 1.00x-3.51x -- so ~2.2x of the per-step wander is the estimator; separately, the estimator is now perfectly deterministic per state (sd 0.000000 over 20 recomputes), so seeding is not the residual.

**Fix:** Acquire next: replace `_min_partition`'s Fiedler branch with an exact cut (the true minimiser was k=1, a single cell vs the rest, on all 30 steps -- a singleton scan is O(n^2) and was exact everywhere I measured; Stoer-Wagner is the general fallback), then re-run --verify and count how many UNSTABLE verdicts survive.

## F3 · Economic-Resource

### `pareto` · high

bench_v2.py:226-236 (Fiedler sign-cut, the branch every default 256-cell run takes) is strictly dominated on BOTH axes by the min row-sum: on 18/18 debiased MI matrices at n=8/12/16 the row-sum equals the exhaustive minimum EXACTLY (and Stoer-Wagner confirms it), while the sign-cut overshoots 1.78x-3.87x, and it costs 4.82 ms vs 0.027 ms at n=256 (178x more expensive to be 2.5x wrong); secondarily ENGINE_REGISTRY["MitosisEngine"] (bench_v2.py:1433) is byte-identical to CONTROLS[0] BenchEngine and has no mitosis, spending ~14 of the 172 measured engine-minutes re-deriving the reference control under a false name.

**Fix:** Replace both _minimum_partition branches with Stoer-Wagner (exact for non-negative weights, 0.04-0.58 ms at n<=20, 431 ms at n=256) or min row-sum with S-W as certification, and drop or rename the MitosisEngine row.

### `landauer` · medium

One Phi read costs 712 ms at 256 real rows = 18 process steps (measured), almost all of it the O(n^2) debiased MI matrix at 4 histogram2d calls per pair (1984 at n=32, ~8192 at n=256) — and >99% of that purchased information is then discarded: only one scalar cut (the 2.5x-wrong one) and the sd of the entries reach `phi`, while `total_mi` is computed at bench_v2.py:156 and never used in the returned value. The debias itself is well spent (removes a 0.93-bit bias at N=128, leaving 0.024±0.038 against a 0.61-bit signal); the waste is entirely in the reduction that follows.

**Fix:** Keep the MI matrix but spend it — feed the exact cut, and either drop the unused `total_mi` sum or report the full cut spectrum that the matrix already paid for.

### `info-budget` · high

A --verify at gate defaults measures 273 min (172 engines + 101 controls; one (engine,seed) sweep = 172.3 s at 256c). The controls run first and the void set is known at bench_v2.py:2174, yet if ANY condition voids, bench_v2.py:2266-2270 blocks deployment regardless of engine scores — so 172 of 273 minutes buy information that cannot change the verdict. The allocation is inverted elsewhere too: commit 937c68b bought a 5x multiplier on the whole gate to attack verdict variance that docs/min-partition-is-not-minimal.md then located in the estimator, while PERSISTENCE still decides on ONE final Phi draw (bench_v2.py:1806) whose own step-to-step sd is 11% of mean (measured over 100 consecutive steps at 256c) against a 20% margin — averaging the last 3 draws costs 2 extra Phi reads = 1.4 s = 3.6% of that condition's 40 s and cuts the decision statistic's SE by 42%.

**Fix:** Short-circuit the engine loop when the void set is non-empty, and move the unspent 3.6% (3-draw final average, exact cut) onto the decision statistic instead of the 5x seed multiplier.

### `optimal-transport` · medium

The restore map the axes rely on does not reach the state it names: pairfield_engine.py:170-171 `set_hiddens` writes A only, so `_three_axes`'s _restore/_set_state leaves G and both coupling matrices where the previous probe left them — measured, two identical replays from the same `base` differ by |ra-rc| = 0.0709 (BenchEngine: 0.000000) with G having drifted 7.2476. That residual is the denominator of the response axis: PairField reads response = 29.95 under the gate's partial restore vs 1e6 under a complete one, a 33,390x understatement. No verdict flip at 32c, but the _CEAdapter docstring (bench_v2.py:1402-1410) records this exact class already producing a false 1.11 reading for an engine whose true ratio is 7,000-17,000.

**Fix:** Give every engine a full snapshot/restore (hiddens + coupling + per-side state) and have _three_axes use it, or make set_hiddens raise when it cannot restore the whole state.

### `dynamic-programming` · medium

`_with_axes` (bench_v2.py:2047-2057) builds a fresh probe and recomputes `_three_axes` once per condition — 8.96 s each at 256c, so 4 of the 5 are redundant: 35.8 s per sweep, 21% of every sweep, 57 min of the 273-min run. They are genuinely the same subproblem because line 2054 passes NO `drive` argument, contradicting the docstring at 1472-1478 that says the axes are measured under the condition's own input (I checked all 7 call sites; only line 1767 passes one). The same non-memoisation is repo-wide: `class PhiIIT` is copy-pasted into 40 files, 14 still compute `total_mi - min_partition_mi` and exactly 1 (bench_v2) carries this session's correction; and the n<=8 exhaustive branch (line 217) enumerates both a mask and its complement, evaluating every distinct cut exactly 2.00x.

**Fix:** Compute the axes once per (factory, cells, dim, hidden, seed) and cache — or pass each condition's real drive so the 5 recomputations are actually different measurements; range the exhaustive loop to 2**(n-1); make the 40 PhiIIT copies import one module.

## F8 · Coverage-Consistency

### `axis-coverage` · high

All 7 negative controls are generic corpses; none is a self-ablation of the engine under test — docs/pairfield-persistence-failure.md:217 measures PairField at strength=0.000 (two populations that never touch) scoring 4/5, better than the shipped 0.03, and no axis in bench_v2's gate can see that.

**Fix:** Add a per-engine ABLATION arm to bench_verify_audit.CONTROLS (mechanism knob → 0, built from the same factory) and void any condition the ablation clears, exactly as for DEAD/SCRAMBLE.

### `axis-coverage` · high

CLAUDE.md:171-173 claims every axis baseline is measured from the population's own null, but 3 of 4 bars are hand-picked constants — bench_v2.py:1647-1649 `integration > 0.001 and response > 1.5` and `change > 0.001`; only `identity` gets a measured derangement floor.

**Fix:** Derive integration/response/change floors from the same collapsed+permuted nulls already computed for identity, or state in CLAUDE.md that three of the four bars are constants.

### `axis-coverage` · medium

Experiment 2's corpse is half-strength: bench_strange_loop.py:75 permutes only `self.eng.A.hiddens`, leaving G's cell identity intact, while the gate's ScrambleEngine (bench_verify_audit.py:74-86) permutes the whole population — so the doc's claim 'the pair with cell identity destroyed' overstates the control.

**Fix:** Permute G's rows in the same step (and re-scramble after the field update), or rename the arm A-SCRAMBLE.

### `cross-tool-consistency` · high

40 copies of `class PhiIIT` exist; on one fixed 12x64 tensor the 39 legacy copies all return exactly 4.908425 while bench_v2's (the only one given debiased MI + min-cut this session) returns 0.281 — 17.5x apart under one class name; 24 further files `import phi_rs`, which is not built in this checkout.

**Fix:** Extract one PhiIIT into a module (e.g. phi_core.py), import it everywhere, and delete the 39 copies; add a test asserting phi_core and phi-rs agree on a fixed matrix.

### `cross-tool-consistency` · high

README.md:423 publishes NarrativeEngine 5/5 DEPLOYABLE at '32 cells · hidden 128 · 5 seeds'; re-running the five VERIFICATION_TESTS at 32c/64d/128h over VERIFY_SEEDS gives per-seed [5,4,4,4,5] — PERSISTENCE fails on seeds 43/44/45, so the table's only DEPLOYABLE verdict does not reproduce.

**Fix:** Re-run `bench_v2.py --verify --cells 32` and regenerate the README table from its output, recording dim/hidden/commit alongside it.

### `cross-tool-consistency` · medium

Doc/code drift on the control layer: CLAUDE.md:164 and README.md:404 both name 6 controls (LINEAR missing) and CLAUDE.md:166-167 claims '31 passed', while bench_verify_audit.CONTROLS[1:] has 7 and `pytest tests/test_gate_controls.py --collect-only` reports 36 collected; README.md:417 says '7 controls' and README.md:773 still says '7조건 검증' after two conditions were retired.

**Fix:** Have the docs cite counted values (len(CONTROLS)-1, len(VERIFICATION_TESTS), pytest's collected count) instead of hand-written numbers, the same fix already applied to the --verify header in fc67804.

### `unowned-load-bearing` · high

`set_hiddens` is an unowned partial-restore protocol that the axes depend on: pairfield_engine.py:170 restores side A only, so G and both coupling matrices keep advancing between the undisturbed and disturbed runs — measured with kick=0 (no perturbation at all) PairField's 'others moved' reads 0.046-0.079 against 0.19-0.30 with the real kick (16-27% of the signal is self-drift), while BenchEngine's sham is exactly 0.00000; after dividing by |kick|~4 the drift alone is ~0.015, 15x the 0.001 integration bar.

**Fix:** Make set_hiddens restore the engine's full state (A, G, both couplings, step counters) or have _three_axes deep-copy/rebuild the engine, and add a kick=0 sham assertion that the ripple is 0.

### `unowned-load-bearing` · high

The deployment-blocking layer has an unlabelled off switch and no automation: `bench_v2.py --verify --no-controls` (bench_v2.py:2465) leaves `voided` all-empty so bench_v2.py:2266 skips the GATE VOID branch and prints DEPLOYABLE with no line saying controls were skipped; and nothing runs tests/test_gate_controls.py — there is no .github/workflows and no hook in .harness/.

**Fix:** Print 'CONTROLS SKIPPED — verdicts uncertified' in the verdict block whenever with_controls is False, and wire `pytest tests/test_gate_controls.py` into a CI job or commit hook.

### `unowned-load-bearing` · medium

Two registry rows name things that do not run: bench_v2.py:1433 maps 'MitosisEngine' to a bare BenchEngine (the same object as CONTROLS[0] 'REAL') though mitosis.MitosisEngine exists and is imported by consciousness_meter.py:461, and bench_v2.py:1418-1421 silently falls back to BenchEngine if consciousness_engine fails to import — so 'ConsciousnessEngine' rows can be the base class.

**Fix:** Point MitosisEngine at mitosis.MitosisEngine (or drop the row), and let _make_ce raise instead of falling back so a missing deployed engine is a failure, not a 5/5.

### `parallel-fanout` · medium

The gate is fully serial on a 10-core box: 5 conditions x (12 engines + 7 controls) x 5 seeds = 475 independent factory-built runs with no multiprocessing/concurrent import anywhere in bench_v2.py, bench_verify_audit.py or tests/test_gate_controls.py, and pytest-xdist is not installed; at the measured 7.7 s per engine-seed at the 256-cell default that is ~60 min wall time where a process pool would give ~8 min.

**Fix:** Wrap the (engine, condition, seed) loop in a ProcessPoolExecutor (each run is already seed-determined and independent) and add pytest-xdist so the enforcement test runs with -n auto.

### `landscape` · medium

The Φ search never left one estimator family (sum of pairwise histogram-MI minus a partition): the candidate ladder is sign-cut → sweep → exhaustive, and the teammate's own table in docs/min-partition-is-not-minimal.md:95-98 shows the sweep still ranks IDENTICAL above INDEPENDENT by 10,886x — the direction defect is a property of the family, and no doc enumerates KSG/copula MI, transfer entropy, Φ*, IIT-4.0 φ, or decoder-based alternatives.

**Fix:** Add one paragraph enumerating at least three out-of-family estimators and score each on the IDENTICAL-vs-INDEPENDENT construction before spending more on the partition heuristic.

### `landscape` · medium

Experiment 2 declares an architecture-level impossibility ('cannot eat its own output') having swept only repulsion strength (0.01, 0.003): the feedback gain is unswept — bench_strange_loop.py:122 feeds back `out.detach()` raw, measured |out|=14.07 at step 0 against the open arm's drive norm ~0.61 (23x mismatch, compounding ~9x/step) while |h|max stays 1.02, and PairFieldEngine's guard (pairfield_engine.py:155-158) watches |h| only, so it cannot see the output blow-up.

**Fix:** Sweep a feedback gain g in `x = out.detach() * g` with g matched to the open arm's drive norm, and extend the divergence guard to the returned output as well as |h|.

## F2 · Adversarial-Stress

### `adversarial` · high

A HEAP (sync/debate/repulsion all 0, no cell ever touches another) plus a private RNG stream that set_hiddens cannot rewind scores 5/5 on the gate; HEAP alone scores 0/5. bench_v2.py:1600 computes integration as |Δothers|/|kick| with no null subtraction — measured 0.32118 total vs 0.31833 from two IDENTICAL runs (99.1% is nondeterminism, 318x the 0.001 bar, zero interaction).

**Fix:** Subtract the engine's own same-input null from `integration` (run the undisturbed step twice, use ripple−null), and fail the axis when null/signal > ~0.5 instead of scoring the sum.

### `adversarial` · high

pairfield_engine.py:170 set_hiddens restores A only — G's hiddens and BOTH accumulating coupling matrices are never rewound, so `_three_axes`' "same base state, one nudge" probe is not a controlled comparison: two identical process(x) calls from the same restored base diverge by 6.58e-2 on a state norm of 41.2, giving a null integration of 0.01122 = 11x the bar before any kick is applied.

**Fix:** Require engines to expose a full snapshot/restore (state dict, not just the hidden matrix) and have `_three_axes` assert restore-identity — two identical runs must be bit-equal — before reading any axis.

### `adversarial` · medium

The 1e6 saturation at bench_v2.py:1642-1643 makes `response` a determinism detector, not a responsiveness measure: every deterministic engine (REAL, HEAP, CLONE, LINEAR) reads exactly 1000000.00, so CLONE and HEAP clear the response conjunct while an engine with any internal noise reads 2-40.

**Fix:** Cap the ratio at a finite value derived from the population's own null (e.g. clamp to the 99th percentile of the same-input spread) so a corpse and a responder are not both at the ceiling.

### `byzantine` · high

A control that lies by crashing is scored as a control that was correctly rejected: bench_v2.py:2138-2139 and tests/test_gate_controls.py:53 both map any exception to passed=False with no output. Demonstrated — a control class whose __init__ raises TypeError yields 5/5 green in the enforcement suite and voids zero conditions in the gate.

**Fix:** Treat an exception from a CONTROL as a hard error (raise / report VOID-UNKNOWN), not as a rejection; only an engine's crash may count as FAIL.

### `byzantine` · medium

bench_v2.py:1417-1421 `_make_ce` silently returns a plain BenchEngine on ImportError, and the summary still prints `ConsciousnessEngine  DEPLOYABLE`; separately ENGINE_REGISTRY["MitosisEngine"] (bench_v2.py:1433) IS `BenchEngine` — the same class as CONTROLS[0] "REAL" — and bench_v2 never imports mitosis.py, so a named row in the deployment table is the control engine wearing another name.

**Fix:** Let the import error propagate, and make run_verify print `type(engine).__name__` next to every registry row so a name can be checked against the object that ran.

### `edge-chaos` · high

The gate's shipped default is `--cells 256 --dim 64 --hidden 128` (bench_v2.py:2472-2478) while every control result and tests/test_gate_controls.py:33 are pinned at 32/32/64. Measured across scale, SCRAMBLE clears all three axes 0/5 seeds at 32c, 2/5 at 64c, 0/5 at 128c, 0/5 at 256c — non-monotonic, so "the axes reject SCRAMBLE" is a coincidence of scale, not a property.

**Fix:** Run `_run_controls` at the same (cells, dim, hidden) the engines are scored at and make tests/test_gate_controls.py parametrise over at least {32, 64, 256}.

### `edge-chaos` · high

At 2 rows the identity null degenerates: `_derangement(2)` (bench_v2.py:1541) has exactly one derangement, so all 60 draws are identical, std=0 and the floor collapses to a single deterministic (usually negative) value clamped to 1e-6 — and `_CEAdapter` returns 2 rows regardless of the requested cell count, so the deployed ConsciousnessEngine is ALWAYS scored in this regime. At cells=2 SCRAMBLE passes SELF_LOOP.

**Fix:** Refuse to emit an identity verdict when the null has zero spread (rows < 4): return "unmeasurable" and fail the condition rather than falling through to the 1e-6 clamp.

### `edge-chaos` · medium

`torch.manual_seed(sd)` does not give engines a common input stream — each engine consumes a different amount of RNG at construction, so the first drive after seed 42 is +0.018898 for ConsciousnessEngine, +0.149337 for PairField, -0.145754 for BenchEngine. Cross-engine and engine-vs-control comparisons at "the same seed" are confounded with the draw.

**Fix:** Pre-generate the drive sequence from a dedicated generator seeded independently of construction (as bench_strange_loop.py:99 already does with `manual_seed(seed+5000)`), and hand it to every engine.

### `perturbation` · high

The response axis is a cliff, not a proportional measure: the real engine unchanged except for a private noise stream reads response 1000000 → 36.22 at noise 0.001 (0.002% of the state norm 43.4) → 7.17 at 0.005 → 2.08 at 0.02 → 1.22 at 0.05 (axes FAIL), while integration moves the other way 0.065 → 3.10. An infinitesimal nudge moves one axis 4.5 orders of magnitude and pushes the other past every control's value.

**Fix:** Report response as (differing − same)/scale against the engine's own same-input null rather than a ratio, so the statistic degrades smoothly instead of saturating at 1e6 and collapsing to 1.

### `perturbation` · high

SCRAMBLE — the session's founding defect — is rejected by exactly one axis with a 0.3σ margin: identity−floor over 30 seeds at 32c is mean −0.000267, sd 0.000883, max +0.000099; over 60 seeds SCRAMBLE clears all three axes on 4 (seeds 57, 73, 82, 85), p≈0.07 per seed. Its other two axes pass comfortably (integration 0.777, response 5.11).

**Fix:** Widen the identity null (paired: run the same population scrambled through the same 60-step loop, not one derangement of the final state) and require identity > floor + 3·sd(identity) rather than > floor.

### `perturbation` · medium

The response estimator is asymmetric in time: `differing`=|ra−rb| spans calls 1→2 while `same`=|ra−rc| spans calls 1→3 (bench_v2.py:1637-1641), so any drifting engine gets its ratio halved. Measured same-input null, which must be 1.0: DECOUPLED 0.50, PairField 0.66, ConsciousnessEngine 0.78 — the nominal 1.5 bar is effectively 2.3-3.0 for a stochastic engine.

**Fix:** Measure `same` from an adjacent pair (calls 2→3 or a fresh ra′) so numerator and denominator span the same number of steps.

### `ablation` · high

Remove `_with_axes` and the five conditions carry nothing for a live engine: REAL passes all five raw at 32c, and the raw conditions only reject DEAD/NOISE/LINEAR — HEAP passes 2 raw, SCRAMBLE 2, DECOUPLED 1, CLONE 1. So "5 conditions × all-5-seeds" is 5 correlated re-draws of one test (`_three_axes` on a fresh 50-step probe), not 5 independent gates, and the multi-seed tightening in 937c68b compounds one statistic five times.

**Fix:** Score `_three_axes` once per (engine, seed) and report it as the gate, with the five conditions as separate named evidence — or give each condition a bar it can actually fail for the real engine.

### `ablation` · high

bench_strange_loop.py's recurrence check cannot fire: `real` is the fraction below the 5th percentile of the very set it is computed from, so it is identically 0.0502 for any input. Measured recurrence for a trajectory that repeats a 6-point cycle ten times = +0.0000, an exact period-3 cycle = +0.0000, a closed circle = +0.0035, a trajectory replayed twice = +0.0018 — all below main()'s `rc > 0.01`. The verdict "되먹임이 실제로 작동" is unreachable by construction.

**Fix:** Compute the threshold from the SHUFFLED null (or from lag>=3 pairs of the null) and compare the real fraction against it, and use `<=` so exact revisits at distance 0 are counted.

### `ablation` · medium

Half of PairFieldEngine is invisible to the gate: get_hiddens/set_hiddens expose A only (pairfield_engine.py:167-171), and freezing G at initialisation — the entire reverse population the repulsion field is defined against — costs 0.2 of 5 (gate 4.80 vs 5.00 mean over seeds 42-46 at 32c), while strength=0.0 (A and G never meet) scores 4.20. Separately G's unruled coupling stream is a fixed generator (pairfield_engine.py:88, seed=1): |coupling| sum is 2.364389 on all five VERIFY_SEEDS, so the multi-seed gate does not vary that half at all.

**Fix:** Have get_hiddens/set_hiddens cover the concatenated A|G state (and the coupling matrices), and seed PairSide._gen from the global RNG so the seed sweep actually perturbs G.

### `ablation` · medium

The negative control in bench_strange_loop.py:74 scrambles A only — G keeps full cell identity and keeps driving both the field (ha−hg) and the output — so the "corpse" arm retains half its dynamics; and pairfield_engine.py:184 writes `strength` into the save blob while load() (189-197) never reads it, so a checkpoint saved at strength 0.001 reloads running at 0.03 (measured).

**Fix:** Permute A and G together in Scrambled.process, and read `strength` back in PairFieldEngine.load (or drop it from the blob so the omission is visible).

