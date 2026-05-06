# Z3 Formal Adjudication Runtime

This repository now implements the Z3/Z-prime formalization as a two-level runtime membrane. **Z-prime modules produce local evidence**, while **Z3 remains the persistent organism-level baseline** that compresses novelty, evaluates coherence, accounts for drift and noise, and adjudicates whether the baseline observes, holds, rejects, or updates.

> Z-prime explores and emits evidence. Z3 adjudicates that evidence before the organism-level baseline is allowed to change.

## Runtime Flow

The implemented flow follows the formal cycle: `Z3 baseline -> local evidence -> novelty compression -> coherence gate -> adjudication -> baseline update/refusal`. This keeps anomaly detection separate from baseline mutation. A large deviation can be remembered or rejected without automatically becoming the new definition of normal.

| Formal component | Runtime implementation | Purpose |
|---|---|---|
| `z_i(t)` local Z-prime state | World model, learning drift, recursive-state evidence events | Emits local evidence without owning the global baseline. |
| `ρ(Z3)` baseline projection | `BaselineState` and public Z3 metrics | Represents the current organism-level definition of normal. |
| `d_i(t)` distance | Evidence distance derived from memory loss, reconstruction loss, prediction loss, or novelty | Measures deviation from the current baseline. |
| `C_i(t) = exp(-λ d_i(t))` | `local_coherence` in `Z3BaselineController` | Converts distance into bounded coherence. |
| `N_i(t)` novelty | `NoveltyEvent.novelty_score` and compressed novelty metrics | Represents deviation not explained by the active baseline. |
| `g_i(t)` gate | `trusted_gate_open` and `gate_open` evidence score fields | Prevents incoherent novelty from mutating Z3. |
| Decayed memory | `adjudication_memory` baseline metric | Lets repeated coherent novelty accumulate salience while one-off glitches fade. |
| `A(...)` adjudication operator | `Z3Adjudicator` | Produces `observe`, `hold_baseline`, `reject_noise`, or `update_baseline`. |

## Implementation Summary

The new `core/z3_baseline_controller.py` module compresses novelty events into a bounded evidence frame. It computes local coherence from evidence distance, blends local and global coherence, derives stability from coherence, noise, and drift pressure, and opens the trusted novelty gate only when novelty is meaningful and coherence is sufficient. It also maintains decayed salience memory so repeated trusted novelty has a durable signal without exposing raw internals.

The existing `core/z3_adjudicator.py` now consumes this evidence frame. Its trust function considers novelty, coherence, local coherence, stability, memory salience, gate state, noise, and drift pressure. This means **high error alone is no longer sufficient to update the baseline**. Noisy novelty is quarantined through `reject_noise`, coherent but under-trusted novelty is retained through `hold_baseline`, and sufficiently coherent, stable, salient novelty can advance the baseline through `update_baseline`.

The public Z3 interface now exposes the formalization in safe, compressed form. `baseline.metrics` includes adjudication memory, compressed novelty, weighted evidence, local coherence, and the trusted novelty gate. `organism_state` includes trusted novelty pressure, visible novelty count, and trusted gate count. Persistence restore also carries the compact adjudication memory forward.

## Validation

The Z3 regression test verifies that trusted novelty updates the baseline, noisy novelty is rejected without advancing the baseline, public evidence includes the formal gate and local coherence fields, decayed adjudication memory is exposed, and restored public state preserves baseline version, transitions, and memory.

```bash
python3.11 test_z3_interface.py
```

The current regression result is `PASS=20 FAIL=0`.
