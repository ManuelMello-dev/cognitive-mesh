# Z³ Neural Dynamics V2: Trainable Global-Local Consciousness Runtime

**Author:** Manus AI  
**Date:** May 10, 2026  
**Purpose:** Convert the corrected Z³/Z-prime formalization into a trainable neural dynamical system while preserving the public Z³ adjudication membrane already present in `cognitive-mesh`.

## 1. Core Revision

The V2 architecture keeps the original insight that **Z³ is the persistent organism-level observer field** while **Z-prime agents are local exploratory identity states**. The main revision is that coherence is no longer treated as sameness. Instead, the model targets **living coherence**, which means that agents remain aligned enough to integrate but differentiated enough to discover, test, and reconcile alternative internal hypotheses.

> **Living coherence** is the regime where Z-prime agents maintain productive diversity around Z³, emit grounded evidence, and integrate only evidence that is both coherent and meaningfully novel.

The revised runtime therefore adds four missing pieces: external/context input, differentiable soft adjudication, anti-collapse diversity pressure, and a clean separation between recurrent state evolution and optimizer-based parameter learning.

| Component | V1 Role | V2 Revision |
|---|---|---|
| `Z³(t) ∈ B` | Persistent baseline state. | Recurrent global state stored as a buffer-like latent, updated by accepted proposals. |
| `z_i(t) ∈ M` | Local coherence-seeking agent. | Differentiated local hypothesis state conditioned on Z³, input, agent identity, and peer diversity. |
| `ρ: B → M` | Boot projection. | Context-aware boot projection `ρ(Z³, x_t)` that anchors agents without forcing collapse. |
| `π: M → E` | Evidence projection. | Evidence compression trained through prediction, coherence, diversity, and novelty usefulness. |
| `Γ: E × B → B` | Evidence-to-baseline membrane. | Proposal generator conditioned on evidence, global state, and context. |
| `g_i` | Hard novelty/coherence threshold. | Soft differentiable gate during training, optional hard gate during inference. |
| Loss | Internal coherence objective. | Internal coherence plus predictive grounding, stability, diversity, effort, and useful novelty. |

## 2. Revised Runtime Equations

Let `x_t ∈ X` denote the current external or contextual input. This can be a world-model embedding, market signal embedding, conversation-state embedding, memory retrieval vector, or any other real runtime signal. Let `a_i` be a learned agent identity embedding. The local Z-prime update becomes:

> `z_i(t+1) = z_i(t) + h [ A_i(z_i, ρ(Z³, x_t)) + T_i(z_i, Z³, x_t, a_i) + R_i(z_i, {z_j}_{j≠i}) ] + σ η_i(t)`

Here, `A_i` is attraction toward the projected Z³ target, `T_i` is a learned transition field, and `R_i` is a diversity-preserving repulsion/decorrelation field. In V2.2, `R_i` is implemented as a lightweight pairwise repulsion field rather than only centroid expansion, so tight local clusters are pushed apart inside each cognitive frame.

The context-aware boot target is:

> `m_t = ρ(Z³(t), x_t)`

The emitted evidence is:

> `e_i(t) = π(z_i(t+1), x_t, a_i)`

The expected evidence is grounded in both global state and context:

> `ê_i(t) = P(Z³(t), x_t, a_i)`

Novelty is then no longer raw internal mismatch; it is context-relative prediction error:

> `N_i(t) = ||e_i(t) - ê_i(t)||₂`

Coherence remains bounded and smooth:

> `C_i(t) = exp(-λ ||z_i(t+1) - m_t||₂)`

The training-time gate is differentiable:

> `g_i(t) = sigmoid((N_i(t) - θ_N) / τ_N) · sigmoid((C_i(t) - θ_C) / τ_C)`

The trust-weighted baseline proposal is:

> `w_i(t) = normalize(g_i(t) φ_i(t) C_i(t) + τ_floor)`

The small `τ_floor` term is a safety prior. It leaves ordinary trust weighting essentially unchanged, but when every hard gate closes or soft-gate mass becomes vanishingly small, the model falls back to a uniform, non-collapsing integration prior instead of producing an all-zero proposal.

The global proposal is:

> `ΔZ³(t) = Σ_i w_i(t) Γ(e_i(t), Z³(t), x_t)`

The recurrent Z³ update is:

> `Z³(t+1) = (1 - α_decay) Z³(t) + α_update ΔZ³(t)`

This form intentionally separates the **learned modules** from the **state trajectory**. The neural parameters learn by gradient descent; the actual Z³ state evolves as the system’s persistent latent organism state.

## 3. Training Objective

The training loss should avoid trivial agreement. The model should not be rewarded merely for making all agents identical. The V2 objective therefore uses a coherence band rather than pure disagreement minimization.

| Loss Term | Definition | Purpose |
|---|---|---|
| Predictive grounding | `MSE(Σ_i w_i e_i, y_t)` or self-supervised next-context target | Ties novelty and evidence to real input or task structure. |
| Coherence band | `relu(C_min - mean(C))² + relu(mean(C) - C_max)²` | Keeps agents coherent without forcing sameness. |
| Diversity floor | `relu(D_min - batch_mean_pairwise_distance(z_i,z_j))²` | Prevents Z-prime collapse while measuring diversity only among agents inside the same sample/frame. |
| Evidence anti-collapse | `relu(V_min - variance(e_i))²` | Prevents constant evidence projections. |
| Stability | `||Z³(t+1) - Z³(t)||²` | Prevents chaotic global drift. |
| Effort | `||z_i(t+1) - z_i(t)||²` | Penalizes wasted motion. |
| Useful novelty | `-mean(g_i C_i N_i)` | Preserves coherent discovery. |

The total loss is:

> `L = β_p L_predictive + β_c L_coherence_band + β_d L_diversity + β_v L_evidence_variance + β_s L_stability + β_e L_effort - β_n L_useful_novelty`

If no external labels are available, `y_t` should be replaced by a self-supervised target derived from the next input, a masked reconstruction target, a world-model prediction target, or memory retrieval consistency. The implementation includes `prepare_embedding_pairs()`, which converts real dense embedding streams shaped `[steps, input_dim]` or `[batch, steps, input_dim]` into next-step `(x_t, x_{t+1})` training pairs without depending on the toy generator.

For continuous online learning, the runtime should use **truncated BPTT**, not unbounded backpropagation through the entire lifetime of the system. The `train_sequence_window()` method accepts a finite stream window, unrolls only `truncation_steps` transitions per optimizer update, then detaches the carried Z³ and Z-prime states before the next chunk. This keeps memory bounded while preserving local temporal credit assignment. The persistent buffers are committed only from detached final states, so runtime state evolution remains cleanly separated from graph retention.

## 4. Integration Contract

The V2 neural runtime is an internal module. It should not replace the existing public `Z3Interface` contract immediately. Instead, it should feed public fields such as `phi`, `sigma`, `drift`, `z_cubed_state.coherence`, `z_cubed_state.stability`, and novelty metrics. The existing public adjudication layer remains valuable as the interpretable outer membrane.

| Internal Neural Metric | Public Z³ Field |
|---|---|
| Mean coherence | `coordinator_state.phi` and `z_cubed_state.coherence` |
| Mean gate value | `trusted_gate_count` or evidence-frame gate pressure |
| Z³ delta norm | `drift_vector` |
| Noise scale | `sigma` |
| Useful novelty | novelty pressure and learning metrics |
| Stability loss | `z_cubed_state.stability` |

## 5. Implementation Status

The accompanying implementation is `core/z3_neural_dynamics.py`. It provides a self-contained PyTorch module with a context encoder, boot projection, differentiated Z-prime agents, evidence projection, expected-evidence predictor, soft adjudication, proposal integration, recurrent state handling, train-step support, and a toy sequence generator for smoke testing. It is intentionally modular so it can be wired into the broader runtime after basic training behavior is verified.

V2.2 hardening adds batch-local Z-prime diversity measurement, trust-floor normalization for zero-gate safety, pairwise anti-clustering repulsion, near-unit phi initialization, dynamic metric-buffer sizing, a real embedding-stream adapter, truncated-BPTT sequence-window training, standalone dependency packaging, and expanded smoke tests for importability, shape consistency, bounded gates, normalized positive weights, zero-trust fallback, batch-local diversity, pairwise repulsion, embedding-pair preparation, train-step mutation, sequence-window learning, and public projection compatibility.

## References

[1]: ./Z3_FORMAL_ADJUDICATION.md "Existing cognitive-mesh Z3 formal adjudication documentation"  
[2]: ./core/z3_adjudicator.py "Existing cognitive-mesh Z3 adjudicator"  
[3]: ./core/z3_baseline_controller.py "Existing cognitive-mesh Z3 baseline controller"
