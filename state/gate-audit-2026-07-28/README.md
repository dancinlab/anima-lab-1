# Raw runs behind the gate audit

Every number quoted in `docs/consciousness-gate-audit.md` for this session comes
from a file here. Kept because a measurement whose output lives only in a
scratchpad cannot be checked by anyone else, and because three claims in this
chain were retracted — the retractions are only auditable against the runs that
produced them.

Engine-progress lines (`calibration deferred`, `split_threshold ... was
unreachable`) are stripped; nothing else is edited.

| file | what it establishes |
|---|---|
| `verify_np256.txt` | the 256-cell gate: 5/5 conditions survived the controls, 45/60, 1 DEPLOYABLE |
| `cond_alone_256.txt` | condition-alone leak at the shipping default, 89/175 against 0/175 with the axes |
| `cond_alone_32.txt` | the same grid at 32 cells, 18/35 — first run, 1 seed |
| `band_8seeds.txt` · `band_rest.txt` | the amplitude band at 8 seeds: 0/8, 5/8, 8/8, 8/8, 8/8, 5/8, 0/8 |
| `pathAB_8seeds.txt` | runtime drive paths, 0/8 against 8/8, Fisher p = 0.000078 |
| `tension_traj.txt` | 162/1300 single steps clear the bar, 0/1295 window means |
| `quantile_cliff.txt` | the bar jumps 1.07 → 1.47 between q0.85 and q0.90 and the pass rate goes to 0 |
| `diag_predicts_drive.txt` | max/q90 separates growing drives from stuck ones; q90/mean does not |
| `gap_vs_cells.txt` | q0.90 = max at 2/4/8/16/32 cells — not a two-cell artefact |
| `two_tensions.txt` | both tension definitions give max/q90 = 1.00 on conversation input |
| `vocab_tail.txt` · `repeat_vs_novel.txt` | message variety creates the tail; repetition-vs-novelty was a one-seed fluke |
| `content_seeds.txt` · `direction_only.txt` · `same_norm.txt` | the split-signal arm, and the 8-seed result that retracted it |
| `runtime_amp.txt` · `runtime_paths.txt` | text_to_vector 0.0172 against raw bytes 0.2474 |
| `ce_band.txt` · `rotation.txt` | early band probes, superseded by the 8-seed run above |
