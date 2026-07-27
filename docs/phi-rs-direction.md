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
