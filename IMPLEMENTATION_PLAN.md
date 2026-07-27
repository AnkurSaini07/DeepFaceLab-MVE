# PyTorch Migration: Implementation Plan

Derived from `requirements.md`. Phases map to Section 13's build order; each phase lists concrete
deliverables and its exit criteria (what "done" means before moving on).

## Phase 0 — Repo prep (done)
- Removed AMP/AMPLegacy/Quick96 model implementations (`models/Model_AMP*`, `models/Model_Quick96`)
  and the `ampconverter` CLI subcommand (`mainscripts/AmpConverter.py`, `main.py`) — out of scope
  per Section 2/3, SAEHD-only port.
- Remaining `models/` surface: `ModelBase.py`, `Model_SAEHD`, `Model_XSeg` (XSeg stays; it's the
  masking-label tool, unrelated to architecture scope).

## Phase 1 — Foundation: SAEHD model port
- New `torch.nn.Module` implementations for DF-variant encoder/inter/decoder, ported from
  `models/Model_SAEHD` (`leras`-based) semantics — same layer shapes/behavior, native PyTorch.
- Discriminator module (for later PatchGAN loss in Phase 7) stubbed with correct I/O shapes now
  so later phases don't need architecture surgery.
- Tests (Section 11.1): forward-pass shape assertions per module against dummy tensors, CPU-only.
- Exit: shape tests pass on CPU; no GPU dependency.

## Phase 2 — Data pipeline
- Wrap existing PNG-metadata read/write (landmarks, mask polygons, source info) unchanged —
  reuse DFLIMG/`samplelib` parsing rather than reimplementing the format.
- `torch.utils.data.Dataset` + `DataLoader` (`num_workers`, `pin_memory=True`,
  `persistent_workers=True`, `prefetch_factor=4`).
- Tests (Section 11.4): PNG metadata round-trip, landmark parsing, mask extraction against
  checked-in fixture images.
- Exit: loader produces correctly-shaped batches from a small fixture faceset on CPU.

## Phase 3 — Precision (BF16 autocast)
- Wrap forward+loss in `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`; no
  `GradScaler`. Master weights stay FP32.
- Smoke test (Section 11.2): single forward/backward/optimizer step on dummy data, assert no NaN
  loss and gradients flow — runs in FP32 on CPU (autocast is CUDA-only), BF16 path exercised only
  once GPU is available.
- Exit: smoke test passes on CPU; autocast branch code-reviewed but not yet runtime-validated
  (needs GPU — flagged as a Phase 3 follow-up once hardware is available).

## Phase 4 — Alignment upgrade
- Swap in InsightFace (preferred) or MediaPipe Face Mesh landmark detector.
- Automated quality filtering: confidence threshold, frame-to-frame jitter flagging, yaw/pitch/roll
  range filtering, reuse/extend blur-sort.
- Temporal smoothing pass (moving average / Kalman) over landmark sequences for video.
- Two-pass alignment: median reference pose/size per clip, re-align constrained to that reference.
- Exit: run against a sample clip, confirm filtering/smoothing reduces jitter qualitatively;
  covered by data-pipeline unit tests where deterministic (e.g., filtering thresholds on fixture
  landmark sequences).

## Phase 5 — Masking
- Two-mask system: face mask (existing) + occlusion mask (new), combined as
  `face_mask * (1 - occlusion_mask)`.
- Occlusion mask generation: lightweight custom mic detector (few dozen boxed examples) as
  primary; SAM as general fallback; MediaPipe Hands for hand-specific cases.
- Feather occlusion boundary tighter than outer face-mask edge.
- Wire combined mask into training loss path (masking only — reconstruction is Phase 10).
- Exit: combined mask correctly excludes occluder pixels from a dummy loss computation in a
  smoke test; visual spot-check on sample frames.

## Phase 6 — Deduplication / pose-balancing
- Shared pipeline stage applied independently to `src` and `dst`:
  perceptual hashing (near-duplicates) + ArcFace embedding similarity (same-pose-different-pixel
  duplicates) + landmark-based pose clustering (bucketed by yaw/pitch/roll).
- Cap clusters at ~3-5 representative frames (sharpest/best-aligned), not 1.
- Feeds pose-bucket gaps identified for Section 8.2 (missing-pose generation) — dedup and
  pose-balancing share the clustering stage per Section 5.3.
- Exit: run on `dst` first (Section 5.4 priority), then `src`; report cluster count / frames
  retained before/after as a sanity check.

## Phase 7 — Loss functions
- Add to SSIM+L1 baseline: LPIPS (VGG feature space), PatchGAN adversarial, ArcFace identity
  similarity.
- All loss terms respect the combined mask from Phase 5 — occluded pixels excluded everywhere,
  not just primary reconstruction loss.
- Exit: smoke test extended to confirm each loss term is finite and mask-respecting (zero
  gradient contribution from masked-out pixels, verified on a synthetic fully-occluded region).

## Phase 8 — Training loop
- LR schedule: warmup + cosine decay (or one-cycle).
- EMA shadow weights for preview/inference.
- Gradient accumulation for effective batch size beyond VRAM limits.
- Checkpointing by best validation metric (e.g. LPIPS on held-out set), not just latest.
- Validation split: small held-out `dst` slice never trained on.
- Logging: LPIPS, identity similarity, loss curves via TensorBoard/W&B.
- Design (not implement) with `DistributedDataParallel` in mind — avoid singleton/global state
  that would block multi-GPU later.
- Exit (Section 11.5): overfit-one-sample test — single image pair, 50-100 steps on CPU, confirm
  loss decreases and output visually converges to the sample. This is the checkpoint that
  validates full training-loop wiring before any GPU time is spent.

## Phase 9 — Validate on clean-frame majority
- First real GPU training run, on the 60-70% unoccluded frames, using everything through Phase 8.
- Track LPIPS / identity similarity / SSIM per Section 11.7's "primary tracked metrics."
- Exit: confirms base quality is acceptable *before* investing in occlusion-reconstruction
  (Section 8.3 sequencing — this is a hard gate, not a nice-to-have).

## Phase 10 — Mouth-occlusion reconstruction
- Landmark-conditioned generation: feed rendered landmark heatmap / mouth-region mask as auxiliary
  generator input alongside the (possibly occluded) raw image.
- Temporal context: short neighboring-frame window to inform reconstruction of an occluded frame.
- Optional two-stage: separate geometry predictor (from visible context/temporal neighbors) →
  conditioning input to generator, ControlNet-style.
- Applies only to occluded frames; explicitly no effect on the clean-frame majority (Section 8.2
  limitation to set expectations on).
- Exit: qualitative A/B (Section 11.7) between Phase 9 checkpoint and this checkpoint specifically
  on occluded-frame outputs.

## Phase 11 — Re-evaluate
- Compare Phase 10 quality impact against Phase 9 baseline using real output before further
  investment (Section 8.3 step 6) — decide whether to iterate on reconstruction or stop here.

## Phase 12 — Optional / later
- Full-model `torch.compile()` wrap (on top of BF16 autocast, separate benefit per Section 4).
- Gradient accumulation tuning, multi-GPU (`DistributedDataParallel`) implementation.

---

## Cross-cutting: CPU-only test suite (Section 11)
Runs continuously from Phase 1 onward, not a separate phase:
- 11.1 shape tests — Phase 1
- 11.2 smoke test (forward/backward/step, no NaN) — Phase 3
- 11.3 weight-conversion tests — only if/when porting pretrained TF weights (not currently planned
  as a hard requirement; add if a pretrained-weight bridge becomes necessary)
- 11.4 data pipeline tests — Phase 2
- 11.5 overfit-one-sample — Phase 8 exit gate
- 11.6 explicitly NOT attempted on CPU: visual quality, convergence, generalization
- 11.7 LLM qualitative eval harness — can be built in parallel with any phase once sample face
  images exist; not blocking, first real use is Phase 10 exit

## Open questions blocking full commitment (Section 14) — resolve before/at Phase 1 kickoff
1. Confirm GPU model via `nvidia-smi` (4070 Ti 12GB vs SUPER 16GB) — affects batch-size targets
   in Section 12.
2. Does `src` share `dst`'s mic-occlusion rate? Affects whether Phase 10 effort splits evenly or
   weights toward `dst`.
3. Approximate `src`/`dst` frame counts — sizes Phase 6 dedup targets and Phase 2 memory-caching
   feasibility.
4. Is MVE fork the literal starting codebase to port from (layer-by-layer port of
   `models/Model_SAEHD`), or a clean-room reimplementation using it as spec only? This changes
   how Phase 1 is executed (port vs. rewrite) — worth deciding before writing the first module.
