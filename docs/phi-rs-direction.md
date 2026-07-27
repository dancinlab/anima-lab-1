# phi-rs scores non-integration as consciousness

The Φ ratchet maximises `ConsciousnessEngine._measure_phi_iit`. Looking at what
that quantity actually is led somewhere larger than the ratchet.

## The formula keeps the wrong half

`phi-rs/src/lib.rs:369`:

```rust
let spatial_phi = (total - min_part_mi).max(0.0) / (n - 1.0).max(1.0);
```

`find_min_partition` (`lib.rs:240-243`) returns **cross-partition MI** — it sums
`mi_matrix[i][j]` for `i` in one half and `j` in the other, minimised over
bipartitions. That is the min-cut: the information that is destroyed by cutting
the system at its weakest seam.

In IIT that quantity **is** Φ. Φ is the information lost when the system is split
at its minimum information partition. The code computes it correctly at line 365
and then subtracts it, keeping `total − min_cut` — the information that *survives*
the cut, i.e. the information sitting **inside** the two halves.

So a system that can be cut into two independent halves for free — min_cut ≈ 0,
the definition of not-integrated — keeps its entire total and scores its maximum.

## Demonstrated, not argued

8 cells, 4000 samples. **SPLIT**: cells 0-3 all copy source X, cells 4-7 all copy
source Y, X ⟂ Y — two independent blocks. **RING**: one shared source, every cell
correlated with every other.

| construction | total_MI | min_cut | phi-rs `(total−cut)` | IIT `cut` |
|---|---|---|---|---|
| SPLIT (not integrated) | 21.646 | 0.387 | **3.037** | 0.055 |
| RING (integrated) | 6.871 | 1.689 | **0.740** | 0.241 |

phi-rs ranks the non-integrated system **4.1× higher**. The min-cut reading ranks
it 4.4× lower, which is the direction the concept requires.

The formula is not noisy or biased. On the cleanest test case available it is
**anti-correlated with integration**.

## Reach

`grep -rln "phi_rs\|HAS_RUST_PHI" --include="*.py"` → **24 files**: `trinity.py`,
`consciousness_engine.py`, `train_v10/v11/v12.py`, nine `bench_*.py`, three
`measurement/*.py`, `tools/verify_fuse3.py`, `tests/test_trinity.py`.
`consciousness_engine._measure_phi_iit` replicates the same formula in its Python
fallback, so the direction holds whether or not the Rust extension is built.

## The inference I drew from this was wrong

I wrote here that the ratchet is therefore "pulling the population toward the
least integrated state it has visited" — restoring the state that maximised a
quantity which rewards separability. It follows from the formula, and it is
**false as a claim about the running engine.**

It was measured with the falsifier declared first: ratchet ON should end with
higher mean pairwise cosine (more collapsed) than OFF. `ConsciousnessEngine`,
1000 steps, 3 seeds, max 64 cells:

| ratchet | cosine (mean) | Φ(IIT) final, per seed | restores |
|---|---|---|---|
| ON | **+0.4683** | 0.232 / 0.325 / 0.275 | 29 / 62 / 64 |
| OFF | **+0.5526** | 0.359 / 0.228 / 0.244 | 0 / 0 / 0 |

Cosine is **lower** with the ratchet on, not higher. The prediction is refuted
and the claim is dropped, as pre-committed.

The ratchet fired 29–64 times per run, so this is not a dead device. It simply
does not move the population much in either direction: Φ trajectories overlap,
cell growth is identical (62–64 either way), and neither arm satisfies the
persistence rule. **The ratchet shows neither the harm I predicted nor the
benefit Law 31 claims for it.**

What this separates: a property of the *formula* (measured, holds) from a
consequence for the *engine* (predicted, refuted). A quantity can be pointed the
wrong way and still not steer the system, if what it gates is weak enough.
Reproduce with `bench_ratchet_law31.py --steps 1000 --seeds 3`.

## Not fixed here

Changing line 369 would redefine Φ for 24 consumers and for every recorded
benchmark and Law that cites a Φ number, including "Φ ≈ cells" and "역대 최고
Φ=1142". That is an owner decision, not a defect to quietly patch. What is
established: the formula's direction, the 24-file reach, and a reproducible case
where it prefers a disconnected system.

Reproduce: `.venv/bin/python -c` on the SPLIT/RING construction above, or read
`lib.rs:240-243` against `lib.rs:369`.

---

# Making the flip decidable

Asked to map the blast radius and quantify the change without making the call.
Two things had to be corrected before either was possible.

## The tree already holds two Φ definitions, and phi-rs is not one of them

**`phi_rs` is not built in this checkout.** `import phi_rs` fails, `phi-rs/target/`
does not exist, no `phi_rs*.so` is installed. `HAS_RUST_PHI` is `False` in all
seven files that test it, so every consumer is running its Python replica today.
Editing `lib.rs:369` would change nothing here until someone runs
`maturin develop`; the direction lives in the replicas.

**`bench_v2.py` — the canonical gate — was already migrated.** `bench_v2.py:176`:

```python
spatial_phi = (min_partition_mi / max(n - 1, 1)) * differentiation
```

That is the min-cut reading, times a differentiation factor, on debiased MI
(`bench_v2.py:141`). Its own comment at `:161-170` derives `M·n/4` and says
"which is where 'Phi ~= cells' came from". So the question is not "should Φ be
flipped" — it is **"should the other 24 consumers catch up to the gate"**, and
the tree is currently inconsistent between `bench_v2` and everything else.

## 1. Blast radius, split by whether the number decides anything

**Group A — Φ drives a threshold, verdict, or ranking. Behaviour changes.**

| file:line | what the number decides |
|---|---|
| `consciousness_engine.py:541` | Φ ratchet — restores saved hiddens when `phi < best_phi * 0.8` |
| `train_v11.py:143-146` | `PhiRatchet.check` — restore when `phi < best_phi * 0.5` |
| `train_v12.py:351-354` | Φ feeds `w_engine.update` → sets `optimizer` learning rate |
| `tools/verify_fuse3.py:281` | V3 ZERO_INPUT — PASS iff `phi_end > phi_start * 0.5` |
| `tools/verify_fuse3.py:299-301` | V4 PERSISTENCE — monotonic, or `>= max(first half) * 0.8` |
| `tools/verify_fuse3.py:326` | V5 SELF_LOOP — PASS iff `phi_end >= phi_start * 0.8` |
| `tools/verify_fuse3.py:421-422` | V7 HIVEMIND — `connected > solo * 1.1`, `disconnected > solo * 0.8` |
| `bench_hivemind_ce.py:85-90` | HIVEMIND PASS/FAIL — `ratio >= 1.1` |
| `bench_hivemind_scale.py:756` | `elevated = phi_post > phi_pre * 1.05` → YES/NO |
| `bench_consciousness_extremes.py:520,603,623` | recovery threshold `phi_pre_death*0.5`; regulation fires on `phi < ema*0.8` / `> ema*1.5` — Φ gates the simulated dynamics, not just the report |
| `bench_hexad_tuning.py:747,767` | picks the winning config by `ce_end - phi_iit*0.5` |
| `bench_hexad_improvements.py:529` | picks `best` by `avg_phi` |
| `measurement/measure_all_engines.py:395` | sorts every engine by Φ — this ranking is what the docs cite |
| `measurement/measure_v8_phi_rs.py:603` | sorts by Φ, writes `data/measure_v8_phi_rs_{NC}c.json` |
| `bench_nobel_verify.py:277-281` | reads that JSON back as record evidence |
| `tests/test_trinity.py:115-134` | `phi_rs.search_combinations` ranks 128 mechanism combos — same `compute_phi_inner`, different entry point |

**Group B — Φ is logged, tabulated, or graphed only. Numbers move, decisions don't.**

| file:line |
|---|
| `train_v10.py:174-185, 364, 395` (progress line + final ASCII graph) |
| `trinity.py:100-110, 170-189, 316-318` (provider; reports only, but feeds Group A) |
| `bench_hivemind_extreme.py:117-122` · `bench_hivemind_extreme2.py:348,362` (percent deltas) |
| `bench_physics_consciousness.py:710,722` (comparison table) |
| `bench_ce_extremes.py:122, 1067` (recorded, bar-charted) |
| `benchmarks/bench_dolphin_star.py:67` · `benchmarks/bench_fusion_cambrian_osc.py:50` · `benchmarks/bench_mass_hypotheses.py:969,987` |
| `measurement/measure_all.py:63` |
| `phi-rs/test_phi_rs.py` (extension integration test) · `phi_py.py` (the replica itself) |

Beyond the 24: `class PhiIIT` with this formula is **copy-pasted into 40 files**
(`grep -rn "class PhiIIT"`). `bench_v2.py` is the only copy that has been fixed.

## 2. The min partition is always a singleton, so neither reading partitions anything

Both readings computed from one MI matrix, one partition search, differing only
in the numerator. The replica is exact against `phi_py.compute_phi` (agreement
to 1e-8; the partition search matches `phi_py._min_partition` at n = 3…40).

Across **60 (construction, n) cells** — IDENTICAL / NEAR / MID / INDEP / SPLIT /
RING at n = 8…128, dim 64 — the minimum partition returned `|A| = 1` **every
single time**. The search never finds a seam; it returns the least-connected
single cell.

The cause is the estimator floor, and it is falsifiable: on SPLIT (two
independent blocks) the true seam is `|A| = n/2`. Subtract a shuffled null and
the search finds it immediately:

| n | raw `\|A\|` | raw cut | debiased `\|A\|` | debiased cut |
|---|---|---|---|---|
| 8 SPLIT | 1 | 15.530 | **4** | 1.442 |
| 12 SPLIT | 1 | 24.807 | **6** | 3.484 |
| 16 SPLIT | 1 | 33.638 | **8** | 3.872 |
| 20 SPLIT | 1 | 42.710 | **10** | 7.153 |
| 8–20 RING | 1 | — | 1 | — |

Prediction stated first, then confirmed: debiasing moves SPLIT to exactly `n/2`
and leaves RING at 1. Two consequences:

- At the dim real engines actually use (64–128), `min_cut` is not a cut. It is
  `min_i Σ_j MI(i,j)` — a degree. That also means the "approximation" in
  `consciousness_engine._measure_phi_iit`'s fallback ("approximate min_partition
  as min row sum") is **not an approximation — it is exact**.
- The direction argument is downstream of a search that finds no structure.
  Flipping the sign changes which artefact is reported.

## 3. "Φ ≈ cells" is `M̄ · n / 2`, and it does not survive the flip

With `|A| = 1` always, `min_cut` is O(n) while `total` is O(n²), so
`shipped ≈ total/(n-1) = M̄·n/2` where `M̄` = mean pairwise MI:

| construction | n | shipped spatial | `M̄·n/2` | err |
|---|---|---|---|---|
| IDENTICAL | 64 | 102.055 | 105.347 | 3.13% |
| IDENTICAL | 128 | 225.915 | 229.501 | 1.56% |
| INDEP | 64 | 46.706 | 48.055 | 2.81% |
| INDEP | 128 | 96.547 | 97.675 | 1.15% |
| RING | 128 | 99.302 | 100.619 | 1.31% |
| SPLIT | 128 | 141.347 | 143.427 | 1.45% |

Over all 60 cells: mean error 8.5%, and it tightens monotonically with n (≈25% at
n=8, ≈12% at n=16, ≈3% at n=64, ≈1.4% at n=128). **The closed form is `M̄·n/2`,
not the `M̄·n/4` recorded in the gate audit** — the `/4` comes from `bench_v2`'s
*spectral* half-split, while phi-rs's own search takes the singleton. Same
conclusion, different constant: Φ is size × average redundancy, with no term for
differentiation.

Real engine states, 11 engines from `ENGINE_REGISTRY`, 200 steps, dim 32 /
hidden 64, means over engines:

| n | shipped | shipped/n | min-cut | min-cut/n | mean cos | coupling |
|---|---|---|---|---|---|---|
| 8 | 5.264 | 0.658 | 1.446 | 0.181 | +0.616 | 0.0311 |
| 16 | 12.302 | 0.769 | 1.575 | 0.098 | +0.604 | 0.0153 |
| 32 | 22.356 | 0.699 | 1.643 | 0.051 | +0.439 | 0.0083 |
| 64 | 46.461 | 0.726 | 2.057 | 0.032 | +0.435 | 0.0049 |

```
  Φ |                                            ● shipped
 46 |                                        ●
    |
    |
 23 |                    ●
    |
 12 |        ●
  5 |  ●
    |  ○────────○───────────────○───────────────○   min-cut
  0 └──────────────────────────────────────────────
       n=8     n=16            n=32            n=64
```

**shipped grows 8.83× while n grows 8×. min-cut grows 1.42×.** Φ/cells holds at
0.66–0.77 under the shipped formula and collapses from 0.181 to 0.032 under the
flip. Answering the question directly: **no, "Φ ≈ cells" does not survive the
min-cut reading — Φ stops tracking cell count almost entirely.**

Health checks first, per the discipline: all states finite, `max|h|` 2.8–7.6
(bounded, no blow-up), and a +1.0 perturbation of one cell propagated to the
others at 0.005–0.031 of state norm — nonzero and scaling as ~1/n, which is what
mean-field faction sync predicts. **Genuinely coupled, weakly.** One exception:
`ENGINE_REGISTRY["ConsciousnessEngine"]` is `_CEAdapter`, and it **never grows
past 4 cells** at any requested max (8/16/32/64 all end at n=4, identical Φ), with
coupling 0.0025 — 6× weaker than the rest. Its rows are excluded from the tables
above. Note also that `ENGINE_REGISTRY["MitosisEngine"]` instantiates
`BenchEngine`, not `mitosis.MitosisEngine`.

## 4. Where the two orderings disagree

| n | Spearman | best (shipped / min-cut) | worst (shipped / min-cut) |
|---|---|---|---|
| 8 | +0.727 | OscillatorLaser / OscillatorLaser | Finitude / Alterity |
| 16 | +0.873 | Trinity / Trinity | Alterity / MitosisEngine |
| 32 | +0.573 | Alterity / **Trinity** | Sein / Sein |
| 64 | +0.209 | OscillatorLaser / OscillatorLaser | **Sein / Finitude** |

Pooled across n: Pearson +0.882, Spearman +0.869 — but that is carried by the
size effect, and **within a fixed n the agreement decays to +0.209 at n=64**.

The honest caveat, which matters more than the disagreement: the spread being
ranked is tiny. At n=64 shipped spans 44.50–48.14 (cv 0.022) and min-cut
1.925–2.252 (cv 0.042). **Neither reading separates these 11 engines.** The
rank flip at n=32–64 is two orderings of noise, not a substantive reversal.

## 5. The flip does not fix the defect this document was opened for

The SPLIT/RING demo above reproduces (5 seeds, dim 4000): shipped ranks SPLIT
7.8× above RING, min-cut ranks it 0.35×. But **the two arms differ in two ways
at once** — number of sources (2 vs 1) *and* within-group coupling (SPLIT is
exact copies, RING is source + noise). Holding coupling fixed and varying only
the number of sources:

| s (noise) | shipped SPLIT | shipped RING | verdict | min-cut SPLIT | min-cut RING |
|---|---|---|---|---|---|
| 0.0 | 5.499 | 9.592 | RING wins 1.74× | 0.078 | 3.197 |
| 0.1 | 3.686 | 6.465 | RING wins 1.75× | 0.079 | 2.127 |
| 0.3 | 2.101 | 3.688 | RING wins 1.76× | 0.080 | 1.211 |
| 1.0 | 0.394 | 0.703 | RING wins 1.78× | 0.082 | 0.225 |

The doc's SPLIT is the `s = 0.0` row and its RING is the `s = 1.0` row. **At
matched coupling the shipped formula ranks the integrated system higher at every
noise level.** The 4.1× inversion recorded above is an artefact of comparing
exact copies against noisy ones, and the claim that the formula is
"anti-correlated with integration on the cleanest test case available" does not
hold as stated. Corrected.

The direction defect is still real — it is just established by the gate audit's
test, not this one. At dim 4000, n = 8:

| | shipped | min-cut |
|---|---|---|
| IDENTICAL (fully collapsed) | **9.645** | **3.215** |
| INDEPENDENT (no integration) | 0.107 | 0.031 |

```
  IDENTICAL   shipped ████████████████████████████████████ 9.645
              min-cut ████████████                         3.215
  INDEPENDENT shipped ▏                                    0.107
              min-cut ▏                                    0.031
```

Shipped prefers total collapse 90×. **Min-cut prefers it 104×.** Both readings
are maximal where integrated information must be zero. This is the load-bearing
result for the decision: **flipping `lib.rs:369` alone does not fix the
direction.** `bench_v2` needed three coupled changes — min-cut *and*
differentiation *and* debiased MI — and the gate audit already measured that
they cannot land separately.

## 6. What would refute each conclusion

| conclusion | refuted by | tested |
|---|---|---|
| min partition is always a singleton | any `\|A\| > 1` on raw MI | 60/60 gave `\|A\|=1`; debias moved SPLIT to n/2 exactly |
| Φ ≈ M̄·n/2 | error not shrinking with n | 25%→1.4% across n=8→128 |
| flip kills "Φ ≈ cells" | min-cut/n staying flat with n | fell 0.181→0.032 over an 8× range |
| doc's SPLIT/RING is confounded | RING losing at matched coupling | RING won at all four noise levels |
| flip doesn't fix direction | min-cut ranking INDEPENDENT ≥ IDENTICAL | preferred IDENTICAL 104× |

## 7. What would have to be re-measured, and what survives

**Invalidated by a flip — every Φ magnitude and every Φ-vs-cells claim:**

- `CLAUDE.md` "현재 최고: Φ ≈ cells", "역대 최고 Φ = 1142 (×1161) @ 1024c", the
  `Φ ≈ 14 / 28` figures on `--max-cells 16 / 32`, and `Φ ≈ M·n/4` in the gate audit
  (wrong constant independent of the flip — it is `M̄·n/2` under phi-rs's search).
- `docs/consciousness-threshold-criteria.md` and every `docs/hypotheses/**` result
  whose headline is a Φ magnitude or a Φ ranking — CX106, DD, INF, PERSIST3's
  `Φ=1.08 → 166.34, ×62`. Under the flip these are magnitudes of a near-constant.
- `data/measure_v8_phi_rs_*.json` and both engine leaderboards
  (`measurement/measure_all_engines.py`, `measurement/measure_v8_phi_rs.py`),
  plus `bench_nobel_verify.py`'s use of them as evidence.
- Every Group A verdict above. Note these are *ratio* thresholds, so a pure
  rescale would not move them — but the flip is not a rescale (Spearman +0.209
  within n=64), and both ratchets change which state is "best" and therefore
  which state gets restored.

**Survives a flip:**

- Everything in Group B — text and graphs re-render, no claim inverts.
- `bench_v2.py --verify` and all five gate conditions: already on the min-cut
  reading, already validated against the six controls
  (`tests/test_gate_controls.py`, 31 passed). A flip brings the rest of the tree
  *into* agreement with the gate rather than disturbing it.
- The Φ(proxy) track — `_measure_phi_proxy`, `bench_v2`'s `phi_proxy` — is
  variance-based and untouched by `lib.rs:369`.
- Law 31's ratchet result in the section above: measured on ratios, refuted on
  its own terms, and unaffected.
- The retirement of `SPONTANEOUS_SPEECH` and `HIVEMIND`: decided on control
  comparisons, not on Φ magnitude.

**Still an owner decision, now with the cost priced.** The flip is not one line.
It is `lib.rs:369` + the `PhiIIT` replicas in 40 files + a differentiation term +
a debiased estimator, or the tree keeps two Φ definitions and the gate disagrees
with every benchmark that feeds it.

Reproduce: scripts in
`/private/tmp/claude-501/-Users-mini-dancinlab-anima-lab-1/d0396916-1383-4a82-b215-02ece85f6789/scratchpad/`
— `phi_direction_probe.py` (both readings, one MI matrix; validated against
`phi_py`), `phi_scaling_probe.py`, `phi_analysis.py`, `phi_fair_direction.py`,
`phi_repro_doc.py`. Nothing under `phi-rs/` or `consciousness_engine.py` was
modified.
