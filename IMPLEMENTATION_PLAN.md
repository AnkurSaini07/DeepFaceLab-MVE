# PyTorch Migration: Implementation Plan

Derived from `requirements.md`. Phases map to Section 13's build order; each phase lists concrete
deliverables and its exit criteria (what "done" means before moving on).

## Resolved decisions (Section 14 open questions)
1. **GPU:** RTX 4070 Ti **SUPER (16GB)**. Use the SUPER batch-size targets in Section 12
   (~20-30% higher than the 12GB baseline at a given resolution).
2. **src occlusion rate:** `src` is cleaner / mostly unoccluded, separately sourced from `dst`.
   Phase 10 (mouth-occlusion reconstruction) effort weights toward `dst` per Section 5.4 — `src`
   gets standard face/occlusion masking (Phase 5) but is not a priority target for reconstruction.
3. **Frame counts:** small, ~1-5k frames each for `src`/`dst`. Full in-RAM caching (Section 4) is
   feasible — no need for partial/streaming cache logic. Dedup targets (Phase 6) stay conservative;
   at this scale, over-aggressive deduplication risks cutting into pose/lighting coverage more than
   it saves compute.
4. **Port strategy (expanded — supersedes the original "literal port vs. clean-room" framing):**
   **Clean-room reimplementation**, not a port. Additional directives from the user:
   - Scan the **entire** codebase for TensorFlow/`leras` dependencies, not just `Model_SAEHD` —
     the footprint is much larger than the training model (see Phase 0.5).
   - Remove all outdated pinned dependencies; adopt current library versions across the board.
   - Use the **existing TF/leras code as the source of truth for characterization tests** — capture
     golden input/output fixtures from the current implementation *before* deleting it, so every
     PyTorch replacement can be validated against real prior behavior, not just shape/smoke tests.

## Phase 0 — Repo prep (done)
- Removed AMP/AMPLegacy/Quick96 model implementations (`models/Model_AMP*`, `models/Model_Quick96`)
  and the `ampconverter` CLI subcommand (`mainscripts/AmpConverter.py`, `main.py`) — out of scope
  per Section 2/3, SAEHD-only port.

## Phase 0.5 — Full TF/leras dependency audit (new — do this before any rewrite)
Full repo scan for TensorFlow/`leras` usage (confirmed via `grep`, 2026-07-27):

**The framework itself — `core/leras/`** (all TF1-native, no longer needed once every consumer
below is ported):
- `archis/` (`ArchiBase.py`, `DeepFakeArchi.py`) — the encoder/inter/decoder architecture defs
- `layers/` (`Conv2D`, `Conv2DTranspose`, `DepthwiseConv2D`, `Dense`, `DenseNorm`, `BatchNorm2D`,
  `InstanceNorm2D`, `FRNorm2D`, `AdaIN`, `BlurPool`, `ScaleAdd`, `TLU`, `TanhPolar`, `MsSsim`,
  `Saveable`, `LayerBase`)
- `models/` (`ModelBase.py`, `CodeDiscriminator.py`, `PatchDiscriminator.py`, `XSeg.py`)
- `optimizers/` (`OptimizerBase.py`, `RMSprop.py`, `AdaBelief.py`)
- `initializers/CA.py`, `ops/__init__.py`, `nn.py` (device/session bootstrap)

**Consumers that import TF/leras directly:**
- `models/ModelBase.py`, `models/Model_SAEHD/Model.py`, `models/Model_XSeg/Model.py` — training
  models (Phase 1 / Phase 5 territory, already planned)
- `facelib/FANExtractor.py`, `facelib/S3FDExtractor.py` — landmark/face detection. **Already
  slated for replacement** by InsightFace/MediaPipe per Section 5.1 — this audit just confirms
  they're TF-based and reinforces that Phase 4 fully retires them rather than leaving them as a
  legacy fallback path.
- `facelib/XSegNet.py` — XSeg mask-inference network. Feeds face-mask generation; needs a PyTorch
  replacement as part of Phase 5 (or Phase 4/6's face-parsing upgrade, Section 6.4).
- `facelib/FaceEnhancer.py` — optional post-processing face enhancer. Not mentioned in
  requirements.md; lowest priority, candidate for replacement with a modern PyTorch
  super-resolution/enhancement model or removal if unused in the target workflow.
- `mainscripts/Extractor.py` — orchestrates FAN/S3FD/XSeg extraction (Phase 4 dependency).
- `mainscripts/Merger.py` — **inference/merge pipeline** (loads a trained model + blends into
  video). Not explicitly in requirements.md's build order, but required for an end-to-end usable
  result post-training. Added as **Phase 12a** below.
- `mainscripts/FacesetEnhancer.py`, `mainscripts/Sorter.py`, `mainscripts/XSegUtil.py`,
  `mainscripts/dev_misc.py` — auxiliary tools; port opportunistically, not on the critical path to
  a trained model.
- `main.py` — calls `nn.initialize_main_env()` at startup; needs updating once `leras` is gone.

**Dependency files to modernize** (`environment.yml`, `requirements-cuda.txt`,
`requirements-colab.txt`): currently pin `python=3.7`, `tensorflow` (TF1-style usage),
`numpy==1.19.3`, `opencv-python==4.1.0.25`, `scipy==1.4.1`, `h5py==3.1.0`, `tf2onnx==1.9.3`,
`Flask==1.1.1`, `flask-socketio==4.2.1`, `Jinja2==3.0.3`, `werkzeug==2.0.2`, `itsdangerous==2.0.1`
— all several years stale. Replace with: current `torch`/`torchvision` (CUDA 12.x build for Ada
Lovelace), current `numpy`/`opencv-python`/`scipy`/`scikit-image`, drop `tensorflow`/`tf2onnx`
entirely (ONNX export, if still needed, goes through `torch.onnx.export`), update Flask stack to
current compatible versions, drop `crc32c`/`h5py` if nothing in the ported code still needs them
(audit at removal time — `h5py` may be used for a save format that's being replaced anyway).

- Exit: a written inventory (this list) confirmed against the live repo — done — plus a decision
  per consumer of "replace in Phase X" vs. "port opportunistically" vs. "drop" (captured above).
  No code changes yet; this phase is audit-only.

## Phase 1 — Foundation: SAEHD model, clean-room PyTorch reimplementation
- **Before deleting anything:** run the current TF/`leras` DF-variant SAEHD model
  (`models/Model_SAEHD/Model.py` + `core/leras/archis/DeepFakeArchi.py`) on fixed-seed dummy
  inputs and capture golden output tensors (encoder/inter/decoder outputs, mask outputs) to disk
  as characterization-test fixtures. This is the reference the new implementation is checked
  against — capture it while the TF code still runs, before Phase 0.5's removal happens.
  - **Done** for the DF-variant core graph: `tests/characterization/capture_saehd_fixtures.py`
    builds `Encoder`/`Inter`/`Decoder` (default dims: res=128, e_dims=64, ae_dims=256, d_dims=64,
    d_mask_dims=22) + `UNetPatchDiscriminator` (patch_size=16, base_ch=16) directly from
    `core/leras/archis/DeepFakeArchi.py`, feeds a seeded (42) dummy 128x128x3 input, and saves
    input/encoder_out/inter_out/decoder_rgb/decoder_mask/discriminator_{center_,}out plus a
    metadata.json to `tests/characterization/fixtures/`. Randomly-initialized weights (no
    training happened) so values are only a shape/range reference, not a semantic one — still
    catches gross structural regressions (wrong shape, wrong output range, NaNs) in the PyTorch
    port. Re-run before further TF removal if dims/opts change: this repo's system Python (3.14)
    has neither TensorFlow nor PyTorch; there's a pre-existing `dfl` conda env
    (`/Users/ankurs/miniconda3/envs/dfl`, Python 3.11, TensorFlow 2.19 via `tf.compat.v1`) that
    runs the legacy code as-is — use it for any further TF-side fixture capture (XSeg, FAN/S3FD
    extractors) before those components are removed.
- New `torch.nn.Module` implementations for encoder/inter/decoder (DF variant) and discriminator
  (PatchGAN, for Phase 7), designed idiomatically in PyTorch — not a line-by-line port. Matches
  `Model_SAEHD`'s documented behavior (layer types, downsampling/upsampling structure, output
  shapes) as a functional spec, but implementation is native.
- Characterization tests: new modules' outputs compared against the golden fixtures within a
  documented tolerance (structural/statistical match, not bit-exact — a from-scratch reimplementation
  won't reproduce TF numerics exactly, but shapes, value ranges, and gradient flow should match).
- Tests (Section 11.1): forward-pass shape assertions per module against dummy tensors, CPU-only.
- Exit: shape tests + characterization tests pass on CPU; discrepancies from the golden fixtures
  are understood and documented (expected numerical drift vs. an actual bug), not silently ignored.

## Phase 2 — Data pipeline (done)
- Wrap existing metadata read/write (landmarks, mask polygons, source info) unchanged — reuse
  `DFLIMG`/`samplelib.SampleLoader` parsing rather than reimplementing the format (this layer is
  pure NumPy/OpenCV, not TF-dependent, so it's reused as-is rather than rewritten). **Correction
  to requirements.md's "PNG-embedded metadata" description:** this MVE fork's `DFLIMG.load()`
  (`DFLIMG/DFLIMG.py`) only recognizes `.jpg` — the actual on-disk aligned-faceset format is JPG
  with embedded metadata, not PNG. Doesn't change anything downstream; noted for accuracy.
- `dfl_torch/data.py`: `SAEHDFaceDataset` (wraps `SampleLoader.load(SampleType.FACE, ...)`,
  resizes to target resolution, returns CHW float32 `[0, 1]` image + mask tensors — XSeg mask if
  present, else `LandmarksProcessor.get_image_hull_mask`) + `build_dataloader()`
  (`pin_memory=True`, `persistent_workers`, `prefetch_factor=4` per Section 4).
- Given the small (~1-5k frame) faceset size (resolved decision #3), `cache_in_ram=True` (default)
  eagerly decodes every sample once at construction and holds tensors in memory — recommend
  `num_workers=0` with this, since the point of caching is avoiding repeat decode cost, and
  worker processes would otherwise pickle the cache across process boundaries for no benefit.
- **Found and fixed a real bug while wiring this up**, not just a version-pin issue:
  `facelib/LandmarksProcessor.py` used the `np.int`/removed-in-NumPy≥1.24 alias in 4 places
  (`expand_eyebrows`, `get_image_eye_mask`, `get_image_mouth_mask`, `draw_landmarks`) — this file
  is one of the "reuse as-is, framework-agnostic" components, and it silently doesn't work under
  any current NumPy. Fixed (`np.int` → `int`, behavior-identical). The same deprecated-alias
  pattern (`np.int`/`np.float`) also exists in `facelib/{S3FDExtractor,FANExtractor}.py`,
  `core/imagelib/text.py`, `core/qtex/qtex.py`, `mainscripts/{Extractor,XSegUtil}.py`,
  `XSegEditor/XSegEditor.py` — not fixed yet since those files are either being replaced outright
  (Phase 4's extractors) or aren't on the critical path for Phase 2; tracked here so Phase 0.5's
  "remove all outdated dependencies" sweep doesn't miss them.
- Tests (Section 11.4, `tests/test_data_pipeline.py`): sample count, item shapes/dtype, value
  range (including a real bug caught here — `cv2.resize(..., INTER_CUBIC)` overshoots `[0, 1]`
  near hard edges and wasn't being clamped on the image, only the mask; fixed), mask
  non-degeneracy, cached-vs-uncached equivalence, batch shapes, empty-directory error handling.
  Fixture faceset: `tests/fixtures/faceset/` (3 synthetic images with procedurally generated
  68-point landmarks embedded via `DFLJPG`, checked in; regenerate via
  `tests/fixtures/generate_face_fixture.py`).
- Exit: 7/7 tests pass on CPU (`.venv-torch`); in-RAM cache path verified identical to non-cached.

## Phase 3 — Precision (BF16 autocast) (done)
- `dfl_torch/precision.py`: `autocast_context(device_type)` wraps
  `torch.autocast(device_type=device_type, dtype=torch.bfloat16)`; no `GradScaler` needed (bf16,
  not fp16). Model weights stay FP32; only ops inside the context run in bf16.
- **Correction to the original plan's assumption:** autocast is *not* CUDA-only — PyTorch's CPU
  autocast supports bfloat16 too (via oneDNN), confirmed empirically (`tests/test_training_smoke.py`
  runs `device_type='cpu'` and the generator's `rgb`/`mask` outputs come back as actual
  `torch.bfloat16` tensors inside the context, `torch.float32` outside it). This means the
  autocast code path itself is exercised on CPU, not just structurally present and
  GPU-only-validated as originally planned — the only thing untested without a GPU is
  CUDA-specific Tensor Core performance/numerics on Ada Lovelace, not correctness of the
  autocast wiring.
- Smoke test (Section 11.2, `tests/test_training_smoke.py`): full generator+discriminator forward
  pass inside `autocast_context`, L1 losses (Section 9's baseline reconstruction term; the fuller
  loss stack is Phase 7) + `.backward()` + `optimizer.step()` on dummy data. Asserts: master
  weights are FP32 before training, ops run in bf16 inside the context, loss is finite FP32
  (no NaN/Inf, no scaler needed), every parameter gets a non-NaN gradient, and weights remain
  FP32 after the step (no accidental downcast).
- Exit: 3/3 tests pass on CPU. Actual CUDA/Tensor-Core behavior on the target 4070 Ti SUPER still
  needs real-GPU validation once available — that part genuinely can't be checked here.

## Phase 4 — Alignment upgrade (retires `FANExtractor`/`S3FDExtractor`) (in progress)
- Replace `facelib/FANExtractor.py` + `facelib/S3FDExtractor.py` (TF) with InsightFace (preferred)
  or MediaPipe Face Mesh — full replacement, not a fallback pair, per the Phase 0.5 audit.
- **Detector choice: MediaPipe, not InsightFace** (deviates from requirements.md's stated
  preference — documented in `dfl_torch/alignment.py`'s module docstring). Reasons: (1) MediaPipe
  is explicitly called out in Section 5.1 as robust to partial occlusion, which is this project's
  actual problem (mic-occluded mouth); (2) its model is one self-contained ~3.7MB download
  (`face_landmarker.task`, cached to `~/.cache/dfl_torch/`), vs. InsightFace's ONNX model-zoo
  dependency — friendlier to this CPU-only/no-persistent-GPU dev loop. Revisit if MediaPipe's
  accuracy proves insufficient on real footage; nothing downstream depends on which detector
  produced the landmarks/pose, so swapping later is contained.
  - Note: the older `mediapipe.solutions.face_mesh` API isn't available in the installed build
    (0.10.35, only `mediapipe.tasks` is exposed) — used the newer Tasks API
    (`vision.FaceLandmarker`) instead.
- **Done:** `dfl_torch/alignment.py` — `FaceLandmarkDetector` (478-point landmarks + 4x4 facial
  transformation matrix per detected face), `estimate_pose_from_matrix` (yaw/pitch/roll via
  standard rotation-matrix Euler decomposition — pure math, unit-tested against known rotation
  matrices independent of real face detection), and the quality-filtering predicates:
  `passes_confidence_threshold`, `passes_pose_range`, `compute_landmark_jitter` /
  `passes_jitter_threshold` (Section 5.1's confidence/pose-range/frame-jitter checks).
  `tests/test_alignment.py`: 13 tests — pose-decomposition math is exhaustively tested (6
  parametrized known-angle cases), filtering predicates fully tested, detector wrapper smoke-tested
  (initializes, downloads+caches its model, correctly reports no-face on non-face input).
  **Known test-coverage limitation:** real detection *accuracy* isn't validated — there's no real
  face photo available as a checked-in test fixture (same category of gap as Section 11.6's
  "explicitly not validated without [X]," here X = a real face). Validate qualitatively once real
  footage is available.
- **Not yet done** (left for a follow-up pass): reusing/extending blur-sort, temporal smoothing
  (moving average / Kalman) over landmark sequences for video, two-pass alignment (median
  reference pose/size per clip), and wiring this into `mainscripts/Extractor.py` in place of
  FAN/S3FD. Characterization fixtures from the old TF extractors (for a sanity cross-check, not a
  strict requirement given the detector swap) also not yet captured.

## Phase 5 — Masking (retires `XSegNet`/`core/leras/models/XSeg.py`)
- Two-mask system: face mask (existing) + occlusion mask (new), combined as
  `face_mask * (1 - occlusion_mask)`.
- Port XSeg mask inference (`facelib/XSegNet.py`) to a native PyTorch module — same
  characterization-fixture approach as Phase 1 before the TF version is removed.
- Occlusion mask generation: lightweight custom mic detector (few dozen boxed examples) as
  primary; SAM as general fallback; MediaPipe Hands for hand-specific cases.
- Feather occlusion boundary tighter than outer face-mask edge.
- Wire combined mask into training loss path (masking only — reconstruction is Phase 10).
- Exit: combined mask correctly excludes occluder pixels from a dummy loss computation in a
  smoke test; visual spot-check on sample frames; XSeg PyTorch output checked against captured
  TF fixtures within tolerance.

## Phase 6 — Deduplication / pose-balancing
- Shared pipeline stage applied independently to `src` and `dst`:
  perceptual hashing (near-duplicates) + ArcFace embedding similarity (same-pose-different-pixel
  duplicates) + landmark-based pose clustering (bucketed by yaw/pitch/roll).
- Cap clusters at ~3-5 representative frames (sharpest/best-aligned), not 1. At ~1-5k frames per
  set, keep this conservative — don't dedup away pose/lighting coverage that's already scarce.
- Feeds pose-bucket gaps identified for Section 8.2 (missing-pose generation) — dedup and
  pose-balancing share the clustering stage per Section 5.3.
- Exit: run on `dst` first (Section 5.4 priority), then `src`; report cluster count / frames
  retained before/after as a sanity check.

## Phase 7 — Loss functions
- Add to SSIM+L1 baseline: LPIPS (VGG feature space), PatchGAN adversarial (using the Phase 1
  discriminator module), ArcFace identity similarity.
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
- First real GPU training run (RTX 4070 Ti SUPER, 16GB), on the 60-70% unoccluded frames, using
  everything through Phase 8. Use the SUPER batch-size targets (~20-30% above the 12GB baseline
  in Section 12) as the starting point, then tune from actual VRAM headroom.
- Track LPIPS / identity similarity / SSIM per Section 11.7's "primary tracked metrics."
- Exit: confirms base quality is acceptable *before* investing in occlusion-reconstruction
  (Section 8.3 sequencing — this is a hard gate, not a nice-to-have).

## Phase 10 — Mouth-occlusion reconstruction (weighted toward `dst`)
- Since `src` is cleaner/mostly unoccluded (resolved decision #2), this phase's effort applies
  almost entirely to `dst`. `src` still gets standard occlusion masking (Phase 5) but isn't a
  reconstruction-quality target — there's little occluded `src` data to reconstruct.
- Landmark-conditioned generation: feed rendered landmark heatmap / mouth-region mask as auxiliary
  generator input alongside the (possibly occluded) raw image.
- Temporal context: short neighboring-frame window to inform reconstruction of an occluded frame.
- Optional two-stage: separate geometry predictor (from visible context/temporal neighbors) →
  conditioning input to generator, ControlNet-style.
- Applies only to occluded `dst` frames; explicitly no effect on the clean-frame majority
  (Section 8.2 limitation to set expectations on).
- Exit: qualitative A/B (Section 11.7) between Phase 9 checkpoint and this checkpoint specifically
  on occluded-`dst`-frame outputs.

## Phase 11 — Re-evaluate
- Compare Phase 10 quality impact against Phase 9 baseline using real output before further
  investment (Section 8.3 step 6) — decide whether to iterate on reconstruction or stop here.

## Phase 12 — Optional / later
- Full-model `torch.compile()` wrap (on top of BF16 autocast, separate benefit per Section 4).
- Gradient accumulation tuning, multi-GPU (`DistributedDataParallel`) implementation.

## Phase 12a — Inference/merge pipeline port (new — required for usable output, not in original build order)
- Port `mainscripts/Merger.py` (loads trained model, blends swapped face into target video/frames)
  to use the Phase 1/5 PyTorch modules instead of TF/`leras`. Without this, a trained PyTorch model
  has no way to actually produce output video — it's a hard requirement for "done," even though
  requirements.md's Section 13 build order doesn't list it explicitly (that list stops at training).
- Opportunistic, lower-priority alongside/after: `mainscripts/Sorter.py`, `mainscripts/XSegUtil.py`,
  `mainscripts/FacesetEnhancer.py`, `facelib/FaceEnhancer.py`, `mainscripts/dev_misc.py` — each
  gets ported or dropped based on whether the target workflow actually uses it; not blocking.
- Exit: end-to-end run — trained checkpoint → `Merger` → output video — on a short clip.

---

## Cross-cutting: characterization testing (new, ties Phase 0.5's directive together)
Because Phase 1+ is a clean-room reimplementation rather than a port, correctness risk shifts from
"did I copy the math right" to "does the new implementation behave equivalently to the old one on
real behavior." Mitigation, applied to every TF/`leras` component being replaced (SAEHD
encoder/inter/decoder, discriminator, XSeg, extractors):
1. **Before** removing a TF component, run it on a small fixed set of fixture inputs (checked-in,
   deterministic) and save outputs as golden fixtures (`.npy`/`.pt`, checked into the repo or a
   test-fixtures directory).
2. **After** writing the PyTorch replacement, run it on the same fixture inputs and diff against
   the golden fixtures with a documented, component-appropriate tolerance — exact match isn't
   the bar (different framework numerics, different init), but shape, value range, and
   qualitative structure (e.g. "landmark boxes land on the same face region," "mask segments the
   same area") should hold.
3. Golden fixtures are captured **once**, while the TF code still exists — this is why Phase 0.5
   (audit) and each phase's "before deleting anything" step must happen in that order. If a TF
   component is deleted before its fixtures are captured, the reference is lost.

## Cross-cutting: CPU-only test suite (Section 11)
Runs continuously from Phase 1 onward, not a separate phase:
- 11.1 shape tests — Phase 1
- 11.2 smoke test (forward/backward/step, no NaN) — Phase 3
- 11.3 weight-conversion tests — not applicable under the clean-room strategy (no pretrained TF
  weight loading planned); superseded by the characterization-fixture approach above
- 11.4 data pipeline tests — Phase 2
- 11.5 overfit-one-sample — Phase 8 exit gate
- 11.6 explicitly NOT attempted on CPU: visual quality, convergence, generalization
- 11.7 LLM qualitative eval harness — can be built in parallel with any phase once sample face
  images exist; not blocking, first real use is Phase 10 exit
