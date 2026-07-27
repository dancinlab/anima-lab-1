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

This also explains the ratchet's measured behaviour. `_phi_ratchet_check` restores
whichever state maximised this quantity — that is, the state that was **most
separable into independent halves**. It is not merely circular with respect to
`PERSISTENCE`; it is pulling the population toward the least integrated state it
has visited.

## Not fixed here

Changing line 369 would redefine Φ for 24 consumers and for every recorded
benchmark and Law that cites a Φ number, including "Φ ≈ cells" and "역대 최고
Φ=1142". That is an owner decision, not a defect to quietly patch. What is
established: the formula's direction, the 24-file reach, and a reproducible case
where it prefers a disconnected system.

Reproduce: `.venv/bin/python -c` on the SPLIT/RING construction above, or read
`lib.rs:240-243` against `lib.rs:369`.
