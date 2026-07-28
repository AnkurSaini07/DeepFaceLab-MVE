# PyTorch Migration: Implementation Plan

Derived from `requirements.md`. Phases map to Section 13's build order; each phase lists concrete
deliverables and its exit criteria (what "done" means before moving on).

## requirements.md update (2026-07-28)
The user revised `requirements.md`, adding Sections 14a/14b/14c and refining Section 15
(renumbered from the original Section 14). Summary of what changed and what it triggered:

- **14a. Clean-Room Implementation Guidance** — confirms the clean-room decision already made
  (resolved decision #4 below), plus 4 specific TF1→PyTorch traps:
  1. Tensor layout (NHWC→NCHW transpose must happen CPU-side, in the DataLoader worker) —
     **already compliant**: `dfl_torch/data.py`'s `SAEHDFaceDataset._load_item` does the
     `.transpose(2, 0, 1)` inside `__getitem__`, which runs in the DataLoader worker process (or
     main process if `num_workers=0`), never on-GPU.
  2. Convolution padding asymmetry in TF `SAME` — not applicable, no weight-conversion path
     exists (Section 11.3 already marked not-applicable under the clean-room strategy).
  3. Weight initialization — **acted on**: leras' actual default (verified empirically via the
     `dfl` conda env, not assumed) is `glorot_uniform` for Conv2D/Dense when no initializer is
     explicitly passed, which is what `DeepFakeArchi.py`/`XSeg.py` do. PyTorch's own defaults
     (Kaiming-uniform) differ. Added `dfl_torch/init.py`'s `apply_xavier_init`, applied to every
     `dfl_torch` network (Encoder/Inter/Decoder, discriminator, XSeg) — see updated Phase 1/5
     entries below.
  4. `torch.compile` graph-break guidance — noted for Phase 12, not started yet.
- **14b. Section 8 revision — Latent Temporal Fusion.** Replaces the original two-stage
  "geometry predictor → conditioning" design for Phase 10 with fusing neighboring frames'
  encoder outputs in latent space (temporal conv/attention in the Inter block) before decoding,
  landmark-conditioning becomes a secondary auxiliary input rather than the primary mechanism.
  Phase 10 hasn't started — see its entry below for the updated design to build against.
- **14c. Section 9 revision — Masked-Loss Implementation Detail.** A real, already-shipped bug
  in Phase 7: masking SSIM/LPIPS by pre-multiplying the *input images* doesn't actually exclude
  occluded pixels from a receptive-field-based computation — see Phase 7's entry below for what
  was wrong and the fix.
- **Section 15 (open questions, renumbered from 14):** GPU question reconfirmed (existing RTX
  4070 Ti, still need `nvidia-smi` to confirm 12GB vs. 16GB SUPER — resolved decision #1 below
  assumed SUPER; unchanged). Clean-room approach reconfirmed (matches resolved decision #4).
  Frame-count guidance refined: **2,000-5,000 diverse frames per identity post-dedup is the
  target; datasets under ~1,500 frames should retain more redundancy rather than aggressively
  deduping.** `src`-occlusion-rate question still open in the doc itself, but was separately
  answered by the user earlier in this project (src is cleaner — resolved decision #2 below,
  unchanged).

## Resolved decisions (Section 14 open questions)
1. **GPU:** RTX 4070 Ti **SUPER (16GB)**. Use the SUPER batch-size targets in Section 12
   (~20-30% higher than the 12GB baseline at a given resolution).
2. **src occlusion rate:** `src` is cleaner / mostly unoccluded, separately sourced from `dst`.
   Phase 10 (mouth-occlusion reconstruction) effort weights toward `dst` per Section 5.4 — `src`
   gets standard face/occlusion masking (Phase 5) but is not a priority target for reconstruction.
3. **Frame counts:** small, ~1-5k frames each for `src`/`dst`. Full in-RAM caching (Section 4) is
   feasible — no need for partial/streaming cache logic. Dedup targets (Phase 6) stay conservative;
   at this scale, over-aggressive deduplication risks cutting into pose/lighting coverage more than
   it saves compute. **Refined 2026-07-28:** target 2,000-5,000 diverse frames per identity
   post-dedup; datasets under ~1,500 frames should retain more redundancy rather than aggressively
   deduping (i.e. lower `dfl_torch.dedup`'s `max_representatives`/`hash_max_distance` aggressiveness
   for small facesets — not yet tuned against real data since none exists in this dev environment).
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
- **Weight initialization (Section 14a point 3, added 2026-07-28):** every network
  (`Encoder`/`Inter`/`Decoder` here, plus `UNetPatchDiscriminator` and `XSegNet` in their own
  phases) calls `dfl_torch.init.apply_xavier_init(self)` at the end of `__init__`, matching
  leras' actual (empirically verified) `glorot_uniform`-weights/zero-bias default — see
  `dfl_torch/init.py`'s docstring for how this was verified rather than assumed.
  `tests/test_weight_init.py`: 5 tests checking every network's Conv2d/ConvTranspose2d/Linear
  weights fall within the theoretical Xavier-uniform bound and biases are exactly zero.
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
  resizes to target resolution, returns CHW float32 `[0, 1]` `(warped, target, mask)` tensor
  triples — XSeg mask if present, else `LandmarksProcessor.get_image_hull_mask`) +
  `build_dataloader()` (`pin_memory=True`, `persistent_workers`, `prefetch_factor=4` per Section 4).
- **Random-warp augmentation added 2026-07-28** (was a documented gap in the end-to-end
  orchestration work below until this point): reuses `core.imagelib.warp`'s `gen_warp_params`/
  `warp_by_params` **unchanged** (framework-agnostic, pure NumPy/OpenCV — no reimplementation),
  matching `models/Model_SAEHD/Model.py`'s actual sample configuration exactly: `warped` gets the
  full elastic-grid distortion + shared affine (rotation/scale/translate/flip) augmentation;
  `target` and `mask` get only that same affine/flip part (no elastic distortion) — sharing one
  `warp_params` draw keeps `warped`/`target`/`mask` positionally aligned, only the elastic warp
  differs. `warp_augment=False` collapses `warped` to equal `target` (e.g. for eval/preview).
  Augmentation is deliberately **not** part of the in-RAM cache — only the expensive decode/mask
  step is cached; fresh random params are drawn on every `__getitem__` call, or caching would
  defeat the point of augmentation.
- **Color/noise/blur/downsample/HSV/shadow augmentation added 2026-07-28**
  (`dfl_torch/augment.py`), closing the rest of Section 4's data-augmentation ask. Reuses
  `core.imagelib`'s standalone functions (`LinearMotionBlur`, `shadow_highlights_augmentation`)
  where DFL already factored them out; noise/jpeg/downsample/HSV-shift are reimplemented matching
  `samplelib/SampleProcessor.py`'s exact formulas, since that logic lives inline inside
  `SampleProcessor`'s large `process` method rather than as standalone functions.
  **Confirmed by reading `models/Model_SAEHD/Model.py`'s actual `output_sample_types` config, not
  assumed:** blur/noise/jpeg/downsample apply *only* to the warped input (the target entry never
  sets these) — the target must stay a clean ground truth. HSV shift and shadow apply to *both*
  warped and target with the *same* random draw (both entries pass identical
  `random_hsv_shift_amount`/`random_shadow` values, sharing the per-sample seed) — keeping color
  grading consistent between input and target, only geometry/sharpness/noise differs.
  `SAEHDFaceDataset` gained `random_blur`/`random_noise`/`random_jpeg`/`random_downsample`/
  `random_hsv_shift_amount`/`random_shadow` params (all off by default); `build_dataloader`/
  `build_train_val_dataloaders` forward them via `**dataset_kwargs` (validation loader never gets
  them, matching its existing `warp_augment=False`); `dfl_torch/train.py`'s `train()` and
  `main.py`'s `train_torch` subcommand both expose them end-to-end. 18 new tests (13 in
  `tests/test_augment.py` for the augmentation functions themselves, 5 in
  `tests/test_data_pipeline.py` for the dataset-level wiring/warped-vs-target split) plus 1 new
  `test_train_e2e.py` case exercising all six flags together through the real CLI.
  **Not implemented:** `ct_mode` (color transfer against a reference face from a *different*
  identity's faceset) — needs an additional data source (a random cross-identity sample), a
  materially separate feature from per-image augmentation. Moving any of this to GPU via `kornia`
  (Section 4's other suggestion) also not done — profiling to justify it needs real data/hardware
  anyway, per Section 4's own "profile before further optimization" guidance.
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
  non-degeneracy, cached-vs-uncached decode equivalence (compared pre-augmentation, since warp
  augmentation is randomized fresh regardless of caching), `warp_augment=True/False` behavior,
  fresh-random-params-per-access, batch shapes, empty-directory error handling.
  Fixture faceset: `tests/fixtures/faceset/` (3 synthetic images with procedurally generated
  68-point landmarks embedded via `DFLJPG`, checked in; regenerate via
  `tests/fixtures/generate_face_fixture.py`).
- Exit: 10/10 tests pass on CPU (`.venv-torch`).

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

## Phase 4 — Alignment upgrade (retires `FANExtractor`/`S3FDExtractor`) (done)
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
- **Also done:** temporal smoothing — `smooth_landmarks_moving_average` (centered, shrinking at
  sequence edges) and `smooth_landmarks_kalman` (per-coordinate scalar constant-position filter;
  defaults `process_var=0.5, measurement_var=4.0`, tuned for ~2px detector noise and realistic
  slow head-sway motion — an earlier attempt with much smaller defaults (`1e-3`/`1e-1`) badly
  lagged behind real motion and made things *worse* than unsmoothed input, caught by a test
  comparing smoothed-vs-ground-truth MSE against raw-vs-ground-truth MSE on a synthetic noisy
  trajectory). Two-pass alignment primitives — `compute_landmark_span`,
  `compute_reference_pose_and_size` (per-clip median pose/size), `passes_reference_deviation`
  (outlier-frame filter), `clamp_size_to_reference` (constrains a frame's crop size toward the
  clip reference instead of discarding it — the "constrained re-run" half of Section 5.1's
  two-pass description). 11 more tests added (24 total in `tests/test_alignment.py`).
- **Extraction pipeline completed 2026-07-28** (`dfl_torch/extract.py`, `main.py extract_torch`)
  — a **clean-room, standalone pipeline, not a patch to `mainscripts/Extractor.py`**, consistent
  with this migration's approach throughout: parallel implementation, legacy code untouched until
  ready to fully retire (Phase 0.5). Covers detect → quality-filter → align → save for a
  directory of frame images; `Extractor.py`'s multi-stage subprocessor architecture, debug
  visualization, and video-to-frames step (a plain ffmpeg call, no TF, doesn't need porting)
  aren't replicated.
  - **Real architectural blocker found and resolved, not routed around:** `facelib.
    LandmarksProcessor.get_transform_mat` (DFL's alignment-crop transform) fits a similarity
    transform against a specific 33-point subset of **dlib's 68-point** landmark scheme, but
    `FaceLandmarkDetector` returns 478 points in MediaPipe's own topology — no shared index
    numbering. Every downstream consumer (`get_transform_mat`, `get_image_hull_mask`,
    `samplelib.SampleLoader`) hard-requires the dlib-68 convention, so extraction output was a
    dead end without a correspondence table. **Rather than hand-derive that table from memory**
    (wrong indices would silently misalign every extracted frame — a high-blast-radius failure
    with no real face photo available here to visually catch it), used `WebSearch`/`WebFetch` to
    find and vendor a maintained, MIT-licensed, purpose-built conversion:
    [PeizhiYan/Mediapipe_2_Dlib_Landmarks](https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks)
    — `dfl_torch/alignment.py`'s `convert_mediapipe_landmarks_to_dlib68` (with the source/license
    documented inline). This is an approximation (different underlying model/topology than a
    real dlib detector), but lands in the exact coordinate convention/index scheme everything
    downstream expects, which is what actually matters for pipeline compatibility.
  - **Found and fixed a second real bug while wiring this up**, same class as the earlier
    `np.int` fix: `get_transform_mat` passes a float64 array to `cv2.getAffineTransform`, which
    this repo's modern OpenCV (5.0.0) rejects (`CV_32F` required) — an older opencv-python
    accepted the looser dtype. Fixed with a one-line `.astype(np.float32)` at the call site
    (`facelib/LandmarksProcessor.py`); caught by an integration test that feeds real converted
    landmarks through the actual (unstubbed) `get_transform_mat`, not just a shape check.
  - `dfl_torch/extract.py`: `extract_one_image` (detect → pose-filter → convert landmarks →
    `get_transform_mat` (reused unchanged) → warp crop → save DFLJPG with the same fields
    `Extractor.py` itself writes — `face_type`, `landmarks` in aligned-crop space,
    `source_landmarks` in original-frame space, `source_rect`, `image_to_face_mat`,
    `source_filename` — so output is an ordinary training-ready sample, not a separate format)
    and `extract_directory` (batch over a folder, returns extracted/skipped counts).
  - `main.py extract_torch` subcommand, added to the same TF-init-skip list as `train_torch`.
  - 12 new tests: 4 in `tests/test_alignment.py` for the conversion table/function (structural
    validity of every index, a synthetic-but-distinguishable-coordinates test proving the
    averaging math is correct, and the `get_transform_mat` integration check that caught the
    float32 bug), 8 in `tests/test_extract.py` (a stub detector isolates extraction-pipeline
    logic from real MediaPipe detection accuracy — same limitation as the rest of Phase 4 — plus
    one test with the real detector against the non-face fixture faceset, and, critically, a test
    that extracted output loads correctly through the *actual* `samplelib.SampleLoader` path
    `dfl_torch.data.SAEHDFaceDataset` uses, not just a structurally-plausible DFLJPG check).
  - **Not implemented:** the temporal smoothing / two-pass alignment primitives (already built in
    `dfl_torch/alignment.py`) aren't wired into `extract.py` yet — each frame is processed
    independently. Reusing/extending blur-sort also not done. Characterization fixtures from the
    old TF extractors (a sanity cross-check, not a strict requirement given the detector swap)
    not captured.

## Phase 5 — Masking (retires `XSegNet`/`core/leras/models/XSeg.py`) (mostly done)
- **Done:** `dfl_torch/xseg.py` — clean-room port of `core/leras/models/XSeg.py`'s 6-level U-Net
  (FRN+TLU activations instead of BatchNorm+ReLU, BlurPool anti-aliased downsampling, dense
  bottleneck, skip connections), matching `facelib/XSegNet.py`'s construction
  (`in_ch=3, base_ch=32, out_ch=1`, resolution=256 = 4·2⁶). Golden fixtures captured from the TF
  version via the `dfl` conda env (`tests/characterization/capture_xseg_fixtures.py`) before
  writing the port, same methodology as Phase 1. 11 tests total: `tests/test_xseg_shapes.py`
  (shapes, sigmoid range, `pretrain` skip-zeroing behaves differently from normal mode, full
  gradient flow through the dense bottleneck) + `tests/characterization/test_xseg_against_tf_fixtures.py`
  (shape/range match against the TF fixtures). Also uses `apply_xavier_init` (Section 14a point
  3, see Phase 1) — covered by `tests/test_weight_init.py`.
- **Done:** `dfl_torch/masking.py` — `combine_masks` (`face_mask * (1 - occlusion_mask)`,
  Section 6.1), `feather_mask` (reuses DFL's existing proportional erode+blur convention from
  `facelib.LandmarksProcessor.blur_image_hull_mask`, generalized to any mask), and
  `feather_combined_mask` (feathers face/occlusion independently, occlusion tighter by default,
  per Section 6.2). 9 tests in `tests/test_masking.py` covering the mask algebra, edge-softening,
  and empty-mask handling.
- **Done, hand-specific case only:** `dfl_torch/occlusion.py` — MediaPipe Hands (`HandLandmarker`
  Tasks API, single bundled model download, same pattern as `dfl_torch/alignment.py`'s face
  detector) + `hand_landmarks_to_occlusion_mask` (convex hull per detected hand, dilated since the
  21 skeleton points trace bones, not silhouette). 4 tests in `tests/test_occlusion.py`.
- **Not implemented** (Section 6.3's other two occlusion sources): a custom-trained mic detector
  — needs "a few dozen manually boxed examples" from this project's actual footage, which don't
  exist yet; and SAM as the general point/box-prompted fallback — a much heavier dependency
  (checkpoint + `segment-anything` package) that wasn't worth pulling in before the mic detector
  (the actually-needed case for this footage) has training data. Both are addable later without
  touching `combine_masks`/`feather_combined_mask` — they just need to produce an occlusion mask
  array in the same format the hand detector does.
- **Not yet done:** wiring the combined mask into an actual training loss computation (Phase 7
  territory — loss functions don't exist yet) and `mainscripts/Extractor.py`/inference-time
  integration.

## Phase 6 — Deduplication / pose-balancing (mostly done)
- Shared pipeline stage applied independently to `src` and `dst` (never against each other, per
  Section 5.3 — nothing here pairs the two datasets).
- **Done:** `dfl_torch/dedup.py` — `compute_dhash` (perceptual difference-hash, self-contained,
  no `imagehash` package dependency), `cluster_by_hash` (union-find on Hamming distance),
  `sharpness_score` (wraps `core.imagelib.estimate_sharpness`, Section 5.1's "reuse/extend DFL's
  existing blur-sort" — **caught a real gotcha**: that function expects uint8 `[0, 255]` and
  silently returns `0.0`, not an error, on float `[0, 1]` input, which would have made every
  frame tie on sharpness and broken representative selection; the wrapper converts dtype before
  calling it), `select_cluster_representatives` (keeps the N sharpest per cluster — Section 5.3's
  cap of ~3-5, not 1), `deduplicate` (end-to-end), and `bucket_by_pose` (yaw/pitch bucketing,
  ignoring roll as an in-plane framing detail rather than a distinct pose — shared with Section
  5.2 pose-balancing per Section 5.3's "combine ... as one shared pipeline stage"). 12 tests in
  `tests/test_dedup.py`.
- **Not implemented:** ArcFace embedding similarity (Section 5.3's third near-duplicate signal,
  for same-pose-different-pixel duplicates dHash won't catch) — needs an ArcFace model, same
  category of heavier-dependency deferral as Phase 5's mic detector/SAM (InsightFace model zoo or
  a standalone ONNX checkpoint, not pulled in yet).
- **Not yet done:** actually running this against real `src`/`dst` footage (no real footage is
  available in this dev environment — only synthetic test fixtures) — exit criteria below apply
  once real data exists.
- Exit (pending real data): run on `dst` first (Section 5.4 priority), then `src`; report cluster
  count / frames retained before/after as a sanity check.

## Phase 7 — Loss functions (mostly done)
- **Correction to "SSIM + L1"** (both requirements.md Section 9 and DFL's own docs use this
  shorthand): `models/Model_SAEHD/Model.py`'s actual reconstruction loss is MS-SSIM + *squared*
  error (`tf.square`), not L1/absolute error. `dfl_torch/losses.py`'s `masked_reconstruction_loss`
  matches what the code actually does, documented explicitly (same spirit as the earlier
  PNG-vs-JPG metadata-format correction in Phase 2).
- **Masking convention revised 2026-07-28 by requirements.md Section 14c — a real bug in what
  had already shipped here.** The original implementation pre-multiplied pred/target by the mask
  *before* computing SSIM (mirroring DFL's actual `gpu_target_src_masked_opt =
  gpu_target_src*gpu_target_srcm_blur` convention), then averaged the resulting map over *all*
  pixels. Section 14c's critique, confirmed by working through the math here: zeroing both
  images over the occluded region makes SSIM there compute the similarity of two identical
  all-zero patches — a **fake ssim≈1** ("perfectly matched"), not an excluded region — and that
  fake-perfect score then got blended into the loss average right alongside the real ones,
  systematically *underestimating* the loss for occluded frames instead of excluding the occluded
  region. (This also silently broke the "severity-based downweighting falls out of averaging over
  everything" reasoning from the original implementation — averaging over everything only
  downweights correctly if the excluded region contributes *zero*, not a fake-favorable score.)
  **Fix:** compute the SSIM map (and squared-error map) from the *unmasked* images, then average
  only over the masked-in region — `dfl_torch/losses.py`'s new `masked_mean(x, mask)` helper,
  used by `masked_reconstruction_loss` and `LPIPSLoss`, matching Section 14c's literal
  `(error * mask).sum() / mask.sum()` formula. `ssim` was split into `ssim_map` (full per-pixel
  map, used internally for masking) and `ssim` (whole-image scalar, for unmasked use).
  Known, expected side effect: SSIM's 11×11 window still legitimately blends a few pixels near
  the mask *boundary* (occluded pixels within the window radius of a visible pixel do get a
  small nonzero gradient, from the visible side's windowed statistics depending on them) — that's
  inherent to any windowed metric, not a bug, and `tests/test_losses.py` checks the *interior* of
  the occluded region (safely beyond the window radius) for exact-zero gradient/loss-invariance
  rather than the whole region.
  The GAN/adversarial loss is **not** affected — Section 14c is explicitly scoped to "LPIPS and
  SSIM specifically," and DFL's own code really does feed a mask-multiplied image into the
  discriminator, so `discriminator_gan_loss`/`generator_adversarial_loss` keep the
  mask-the-input-image convention unchanged.
- **Done:** `dfl_torch/losses.py` — `ssim_map`/`ssim` (single-scale, Gaussian-windowed,
  clean-room; DFL's is multi-scale via `tf.image.ssim_multiscale`, simplified here),
  `masked_reconstruction_loss` (MS-SSIM + squared error, correctly masked per above),
  `LPIPSLoss` (wraps the `lpips` package in `spatial=True` mode so it returns a per-pixel map
  that can be correctly masked post-hoc, rather than masking the input images pre-network;
  forces `.eval()` and overrides `train()` to stay frozen even if a parent module calls
  `.train()` on it, per Section 14c's explicit frozen/eval requirement), `discriminator_gan_loss`
  / `generator_adversarial_loss` (BCE-with-logits, matching DFL's actual `DLoss` — **not hinge
  loss**, confirmed by reading `Model_SAEHD/Model.py`'s `DLoss` definition rather than assuming a
  convention) using the Phase 1 discriminator.
- **Not implemented:** identity-preservation loss (ArcFace embedding similarity) — same
  heavier-dependency deferral as Phase 5's mic detector and Phase 6's ArcFace dedup signal.
- Exit: 16 tests in `tests/test_losses.py`. Covers the plan's key exit criterion directly —
  interior-of-occluded-region gets exactly zero gradient and the loss is invariant to that
  region's prediction content (both explicitly excluding the boundary-bleed pixels, see above).
  GAN losses and LPIPS tested for finiteness, directional correctness (discriminator loss lower
  when confidently correct; generator loss lower when fooling the discriminator), gradient flow,
  mask-respecting behavior, and — the new Section 14c requirement — that LPIPS stays in eval
  mode even inside a training wrapper and that `.backward()` never populates `.grad` on its
  frozen parameters.

## Phase 8 — Training loop (done)
- **Done:** `dfl_torch/training.py` —
  - `build_lr_scheduler`: linear warmup + cosine decay (`LambdaLR`-based), `min_lr_ratio` floor
    instead of decaying to zero.
  - `EMA`: shadow weights tracked by state-dict name (decay-weighted running average), `update`,
    `copy_to` (load shadow into a model for preview/inference), `state_dict`/`load_state_dict`
    for checkpointing.
  - `GradientAccumulator`: scales the loss by `1/accumulation_steps` before `.backward()`, only
    calls `optimizer.step()`/`zero_grad()` every N calls to `.step()`.
  - `train_val_split`: seeded random held-out slice (never trained on), `val_fraction` with a
    minimum of 1 sample for small datasets.
  - `CheckpointManager`: `maybe_save` only writes `best.pt` when the given metric actually
    improves (`higher_is_better` flag for SSIM-like vs. LPIPS-like metrics), `save_latest`
    always overwrites `latest.pt` — the "best" and "latest" checkpoints are deliberately
    separate files so a later degraded run can't silently clobber the best one.
  - `TrainingLogger`: thin wrapper around `torch.utils.tensorboard.SummaryWriter`.
  - **DDP-readiness (Section 10's "design, don't implement yet"):** every class here takes the
    model/optimizer it operates on as an explicit constructor/method argument — no module-level
    singleton or global state — so none of this needs to change when the underlying model is
    later wrapped in `DistributedDataParallel`.
- Tests: `tests/test_training.py`, 21 tests covering the scheduler's warmup/decay/floor behavior,
  EMA's convergence properties, gradient accumulation's step-only-every-N-calls behavior, the
  validation split's determinism/coverage/no-overlap, and checkpoint save/skip/load correctness.
- **Exit (Section 11.5) — done:** `test_overfit_one_sample_loss_decreases_and_converges` builds
  the actual `Encoder`/`Inter`/`Decoder` generator, trains on a single fixed image pair for 100
  steps using `masked_reconstruction_loss` + `build_lr_scheduler` + Adam, and asserts the loss
  drops to under 20% of its initial value, keeps improving through the second half of training
  (not just an early cliff), and the final reconstruction's mean pixel error is under 0.1 —
  validating the full training-loop wiring end-to-end on CPU before any GPU time is spent.

## Cross-cutting: end-to-end training orchestration (done, 2026-07-28)
A gap identified after Phase 8: every `dfl_torch` piece (data loading, model, precision, losses,
training-loop utilities) had its own unit tests, but nothing had ever assembled them into one
runnable training script — including, critically, **the full DF-variant SAEHD model itself**
(shared `Encoder`/`Inter` + dual `decoder_src`/`decoder_dst`, the actual face-swap architecture).
Every prior test exercised a single `Decoder` in isolation; nothing had built or tested the
two-decoder assembly that makes DF-variant SAEHD a face swapper rather than a generic
autoencoder. This closes that gap ahead of Phase 9, so Phase 9's real GPU run is exercising code
that's already been proven to compose correctly, not code whose integration is untested.

- **`dfl_torch/model.py` — `SAEHDModel`:** shared `encoder`/`inter`, separate `decoder_src`/
  `decoder_dst`, matching `models/Model_SAEHD/Model.py`'s DF-branch construction exactly (same
  `encoder_out_ch`/`lowest_dense_res` math already used in the Phase 1 fixture-capture script).
  `forward_src`/`forward_dst` for training (reconstruct through the matching decoder);
  `swap(dst_image)` for inference (encode dst, decode with the *src* decoder — the actual face
  swap; only meaningful once trained, not exercised by training itself).
  `tests/test_model.py`: 6 tests, including confirming `decoder_src`/`decoder_dst` share no
  parameters (caught one test-authoring mistake along the way: computing gradients from only the
  RGB output leaves the mask-decoding branch — `upscalem*`/`out_convm`, a disconnected path from
  the shared latent — with no gradient, which is correct decoder behavior, not a bug; the test
  needed both outputs in its loss to check the *whole* decoder).
- **`dfl_torch/train.py`:** orchestrates data loading → `SAEHDModel` → BF16 autocast → masked
  reconstruction (+ optional GAN) loss → LR schedule/EMA/grad accumulation/checkpointing/logging
  into one `train(...)` function plus an `argparse` CLI. Trains on the encoder seeing the
  elastically-warped `warped` image, comparing against `target` (same affine/flip augmentation,
  no elastic distortion) — DFL's actual `warp=True`/`warp=False` sample-pair convention, once
  `dfl_torch/data.py` grew random-warp augmentation (see Phase 2's updated entry above — this was
  a documented gap here initially, closed the same day).
- `tests/test_train_e2e.py`: 3 tests against the checked-in fixture faceset (both src and dst
  point at the same 3-image fixture directory, since there's no separate real dataset) — a full
  run writes valid checkpoints that reload cleanly into a fresh model, a run with `gan_power > 0`
  exercises the discriminator path too, and a longer (150-step) run confirms the loss keeps
  dropping (last third under 92% of the first third's average — deliberately not a dramatic
  threshold: with real warp augmentation every step sees a *different* random distortion of the
  same 3 images, so the toy model has to learn an actual "undo the warp" function rather than
  memorize a fixed mapping, a genuinely harder objective than the pre-augmentation version of
  this test; empirically 10-19% reduction across several seeds at this step count).
- Exit: 130 tests passing total (was 118 before this cross-cutting work; +9 for the SAEHDModel/
  train.py assembly, +3 net for random-warp augmentation added the same day).

### Training workflow completion (done, 2026-07-28)
User explicitly asked to complete the training workflow before touching any of the deferred
"automation" items (real dedup runs, Extractor.py/Merger.py wiring, ArcFace/SAM/mic-detector).
Three concrete gaps closed:

- **Missing losses wired into `train.py`:** `LPIPSLoss` (Phase 7) and `train_val_split`-based
  validation (Phase 8) existed but were never used by the actual training loop.
  - `dfl_torch/data.py`: new `build_train_val_dataloaders` — two `SAEHDFaceDataset` instances
    against the same `samples_path` (train: `warp_augment=True`; val: `False`, since evaluation
    should reflect real unaugmented faces), split by disjoint indices from `train_val_split`.
    Relies on `samplelib.SampleLoader`'s per-path caching returning the same sample ordering on
    both constructions (verified empirically — the second construction hits the cache, no second
    "Loading samples" progress bar) so the same index refers to the same frame in both datasets.
  - `train()` now evaluates on the held-out split every `checkpoint_every` steps and — per
    Section 10's explicit ask — **checkpoints by validation loss, not training loss**: training
    loss can keep dropping from memorization while validation loss stalls, and saving "best" by
    training loss (the original behavior) would miss exactly the overfitting case Section 10
    calls out.
  - `lpips_weight` param (default `0`, so LPIPS's ~230MB network is never loaded unless asked
    for) adds `LPIPSLoss` to both the training loss and the validation metric.
  - **Confirmed by reading `models/Model_SAEHD/Model.py` directly, not assumed:** the adversarial
    loss there only ever applies to `pred_src_src`/`target_src`, despite the internal variable
    name (`D_src_dst_loss`) suggesting otherwise — `dst`'s reconstruction never touches the
    discriminator. `dfl_torch/train.py` already matched this from when GAN support was first
    added; confirmed correct, not changed. (DFL also uses noisy/smoothed real-fake labels for GAN
    stability — not implemented, a minor stabilization detail, noted here rather than silently
    skipped.)
- **Checkpoint resume + preview images:**
  - Checkpoints now include `optimizer`/`scheduler` state (previously only `model`/`ema`/`step`)
    so `--resume-from` actually continues training rather than restarting the optimizer/LR
    schedule from scratch. Discriminator + its optimizer are included too when GAN is enabled.
  - **Found and fixed a real off-by-one bug while wiring resume, not just adding the feature:**
    checkpoints recorded `step` as the *last completed loop index* (0-based) rather than *steps
    completed*. Resuming with `start_step = checkpoint["step"]` and `range(start_step,
    total_steps)` would silently **re-run the last step of the previous run** — caught by a test
    asserting the exact step count after a resumed run, not by inspection. Fixed by recording
    `step + 1` in the checkpoint (the correct "next index to run").
  - `_save_preview`: every `preview_every` steps, runs the model in eval mode on a fixed batch
    and saves a `torchvision.utils.make_grid` PNG (`[src target | src recon | dst target | dst
    recon | dst→src swap]`) to `<output_dir>/previews/step_NNNNNNN.png` — a static-file
    equivalent of DFL's live preview window, since this pipeline has no GUI.
- **`main.py` CLI integration:** new `train_torch` subcommand (`main.py train_torch
  --training-data-src-dir ... --training-data-dst-dir ... --model-dir ...`), argument names
  mirroring `train()`'s parameters.
  - **Found and fixed a real structural blocker, not just added the subcommand:** `main.py`
    unconditionally calls `nn.initialize_main_env()` at the very top of `__main__`, before any
    argument parsing — and that function spawns a subprocess that imports TensorFlow to
    enumerate devices. Every subcommand, including a brand-new pure-PyTorch one, was blocked on
    TF being importable. Fixed with a minimal, targeted change: skip that call specifically when
    `"train_torch" in sys.argv` (checked before any TF-touching import happens) — every other
    subcommand's behavior is unchanged (verified `main.py train --help` still initializes
    normally under the `dfl` conda env). Verified `main.py train_torch --help` and a real
    end-to-end run both work from `.venv-torch` (no TF installed there at all).
- 10 new/changed tests (140 total): `build_train_val_dataloaders` split correctness/disjointness/
  augmentation behavior (`tests/test_data_pipeline.py`), checkpoint state-dict keys, discriminator
  state persistence, LPIPS integration (network-dependent, same skip-if-no-network pattern as
  other detector/LPIPS tests), resume step-count correctness, preview file creation/non-creation.

## Phase 9 — Validate on clean-frame majority
- First real GPU training run (RTX 4070 Ti SUPER, 16GB), on the 60-70% unoccluded frames, using
  everything through Phase 8. Use the SUPER batch-size targets (~20-30% above the 12GB baseline
  in Section 12) as the starting point, then tune from actual VRAM headroom.
- Track LPIPS / identity similarity / SSIM per Section 11.7's "primary tracked metrics."
- Exit: confirms base quality is acceptable *before* investing in occlusion-reconstruction
  (Section 8.3 sequencing — this is a hard gate, not a nice-to-have).

## Phase 10 — Mouth-occlusion reconstruction (weighted toward `dst`) — design revised 2026-07-28
- **Design superseded by requirements.md Section 14b** (not started yet, so this is a plan update
  only, no rework needed). Original Section 8.1 point 3 proposed a two-stage design: a separate
  geometry predictor (estimating mouth shape from visible context/temporal neighbors) feeding the
  generator as ControlNet-style conditioning. **Section 14b replaces this** with a simpler,
  end-to-end differentiable approach — **latent temporal fusion**:
  - Pass the current frame + N neighboring frames (e.g. t-2..t+2) through the shared `Encoder`.
  - Fuse the resulting latent vectors across the time dimension inside the `Inter` block (a
    lightweight temporal convolution or attention mechanism), before passing the fused
    representation to `Decoder`.
  - No separate 3D mesh renderer or standalone geometry-prediction network needed — the network
    learns directly to fill occluded regions using unoccluded latent information from
    neighboring frames.
  - Landmark-conditioning (original point 1) can still be added as an auxiliary input channel
    alongside this, but is no longer the primary recovery mechanism.
  - Implementation implication: `dfl_torch/df_archi.py`'s `Inter` will need a temporal-fusion
    variant (or an alternate `Inter` class) that accepts a stack of encoder outputs across the
    time window rather than a single one — not built yet, noted here for when this phase starts.
- Since `src` is cleaner/mostly unoccluded (resolved decision #2), this phase's effort applies
  almost entirely to `dst`. `src` still gets standard occlusion masking (Phase 5) but isn't a
  reconstruction-quality target — there's little occluded `src` data to reconstruct.
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
