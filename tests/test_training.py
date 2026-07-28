"""
Tests for dfl_torch/training.py (requirements.md Section 10: training loop improvements) and the
Section 11.5 exit criterion for Phase 8: an overfit-one-sample test proving the full training
loop (generator + optimizer + LR schedule + EMA + masked loss) actually reduces loss over
repeated steps on CPU, before any GPU time is spent.
"""
import pytest
import torch
import torch.nn as nn

from dfl_torch.df_archi import Decoder, Encoder, Inter
from dfl_torch.losses import masked_reconstruction_loss
from dfl_torch.training import (
    CheckpointManager,
    EMA,
    GradientAccumulator,
    TrainingLogger,
    build_lr_scheduler,
    train_val_split,
)


# --- LR scheduler ---

def test_lr_scheduler_warmup_reaches_base_lr():
    model_param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([model_param], lr=0.1)
    sched = build_lr_scheduler(opt, warmup_steps=10, total_steps=100)

    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    assert lrs[0] == 0.0
    assert lrs[-1] < 0.1
    assert opt.param_groups[0]["lr"] == 0.1  # exactly at the end of warmup


def test_lr_scheduler_decays_after_warmup():
    model_param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([model_param], lr=0.1)
    sched = build_lr_scheduler(opt, warmup_steps=10, total_steps=100)

    for _ in range(10):
        opt.step()
        sched.step()
    lr_at_warmup_end = opt.param_groups[0]["lr"]

    for _ in range(89):
        opt.step()
        sched.step()
    lr_near_end = opt.param_groups[0]["lr"]

    assert lr_near_end < lr_at_warmup_end
    assert lr_near_end < 0.01  # cosine decay should have brought it close to 0


def test_lr_scheduler_respects_min_lr_ratio():
    model_param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([model_param], lr=0.1)
    sched = build_lr_scheduler(opt, warmup_steps=5, total_steps=20, min_lr_ratio=0.1)

    for _ in range(20):
        opt.step()
        sched.step()

    assert opt.param_groups[0]["lr"] == pytest.approx(0.01, abs=1e-6)


def test_lr_scheduler_raises_if_warmup_not_less_than_total():
    model_param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([model_param], lr=0.1)
    with pytest.raises(ValueError):
        build_lr_scheduler(opt, warmup_steps=100, total_steps=100)


# --- EMA ---

def _tiny_model():
    return nn.Linear(4, 4)


def test_ema_initial_shadow_equals_model():
    model = _tiny_model()
    ema = EMA(model, decay=0.9)
    for name, param in model.state_dict().items():
        assert torch.equal(ema.shadow[name], param)


def test_ema_shadow_moves_toward_model_but_not_all_the_way_in_one_step():
    torch.manual_seed(0)
    model = _tiny_model()
    ema = EMA(model, decay=0.9)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # shift model weights away from the shadow's initial snapshot

    ema.update(model)

    for name, shadow_param in ema.shadow.items():
        model_param = model.state_dict()[name]
        # shadow should have moved toward model_param, but not reached it (decay=0.9 means only
        # a 10% step toward the new value)
        assert not torch.equal(shadow_param, model_param)


def test_ema_converges_toward_model_after_many_updates_with_low_decay():
    torch.manual_seed(0)
    model = _tiny_model()
    ema = EMA(model, decay=0.5)  # fast-converging for the test

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    for _ in range(30):
        ema.update(model)

    for name, shadow_param in ema.shadow.items():
        model_param = model.state_dict()[name]
        assert torch.allclose(shadow_param, model_param, atol=1e-3)


def test_ema_copy_to_loads_shadow_weights():
    model = _tiny_model()
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(5.0)
    ema.update(model)

    target_model = _tiny_model()
    ema.copy_to(target_model)
    for name, shadow_param in ema.shadow.items():
        assert torch.equal(target_model.state_dict()[name], shadow_param)


# --- GradientAccumulator ---

def test_gradient_accumulator_steps_only_every_n_calls():
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    accumulator = GradientAccumulator(accumulation_steps=3)

    stepped_flags = []
    for _ in range(7):
        loss = model(torch.rand(1, 2)).sum()
        accumulator.scale_loss(loss).backward()
        stepped_flags.append(accumulator.step(opt))

    assert stepped_flags == [False, False, True, False, False, True, False]


def test_gradient_accumulator_scale_loss_divides_by_steps():
    accumulator = GradientAccumulator(accumulation_steps=4)
    loss = torch.tensor(8.0)
    assert accumulator.scale_loss(loss).item() == 2.0


def test_gradient_accumulator_rejects_invalid_steps():
    with pytest.raises(ValueError):
        GradientAccumulator(accumulation_steps=0)


# --- train_val_split ---

def test_train_val_split_sizes():
    train_idx, val_idx = train_val_split(1000, val_fraction=0.05)
    assert len(val_idx) == 50
    assert len(train_idx) == 950


def test_train_val_split_no_overlap_and_covers_everything():
    train_idx, val_idx = train_val_split(200, val_fraction=0.1)
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx) | set(val_idx) == set(range(200))


def test_train_val_split_deterministic_with_seed():
    train_a, val_a = train_val_split(500, val_fraction=0.05, seed=7)
    train_b, val_b = train_val_split(500, val_fraction=0.05, seed=7)
    assert train_a == train_b
    assert val_a == val_b


def test_train_val_split_small_dataset_gets_at_least_one_val_sample():
    train_idx, val_idx = train_val_split(10, val_fraction=0.05)
    assert len(val_idx) >= 1


# --- CheckpointManager ---

def test_checkpoint_manager_saves_on_first_call(tmp_path):
    manager = CheckpointManager(tmp_path)
    saved = manager.maybe_save(metric=0.5, state_dict={"step": 1})
    assert saved
    assert (tmp_path / "best.pt").exists()


def test_checkpoint_manager_saves_on_improvement_skips_regression(tmp_path):
    manager = CheckpointManager(tmp_path, higher_is_better=False)  # lower LPIPS is better
    assert manager.maybe_save(metric=0.5, state_dict={"step": 1})
    assert manager.maybe_save(metric=0.3, state_dict={"step": 2})  # improved (lower)
    assert not manager.maybe_save(metric=0.4, state_dict={"step": 3})  # regressed (higher)

    loaded = manager.load()
    assert loaded["step"] == 2


def test_checkpoint_manager_higher_is_better():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        manager = CheckpointManager(d, higher_is_better=True)  # e.g. SSIM
        assert manager.maybe_save(metric=0.5, state_dict={"step": 1})
        assert manager.maybe_save(metric=0.7, state_dict={"step": 2})  # improved (higher)
        assert not manager.maybe_save(metric=0.6, state_dict={"step": 3})  # regressed (lower)


def test_checkpoint_manager_save_latest_always_overwrites(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.save_latest({"step": 1})
    manager.save_latest({"step": 2})
    loaded = manager.load(filename="latest.pt")
    assert loaded["step"] == 2


# --- TrainingLogger ---

def test_training_logger_writes_event_file(tmp_path):
    logger = TrainingLogger(tmp_path)
    logger.log_scalar("loss/train", 1.23, step=0)
    logger.log_scalars({"lpips": 0.5, "ssim": 0.9}, step=1)
    logger.close()
    assert any(tmp_path.iterdir())


# --- Section 11.5 exit criterion: overfit-one-sample ---

def _build_generator(resolution=32, e_dims=8, ae_dims=16, d_dims=8, d_mask_dims=4):
    encoder = Encoder(in_ch=3, e_ch=e_dims)
    encoder_out_res = encoder.get_out_res(resolution)
    encoder_out_ch = encoder.get_out_ch() * encoder_out_res**2
    inter = Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, lowest_dense_res=resolution // 16)
    decoder = Decoder(in_ch=inter.get_out_ch(), d_ch=d_dims, d_mask_ch=d_mask_dims)
    return encoder, inter, decoder


def test_overfit_one_sample_loss_decreases_and_converges():
    """
    Section 11.5: train on a single image pair for ~50-100 steps on CPU; confirm loss decreases
    and the model can memorize the example. Validates the full training-loop wiring (generator +
    optimizer + LR schedule + EMA + masked reconstruction loss) before investing in GPU time.
    """
    torch.manual_seed(0)
    resolution = 32
    encoder, inter, decoder = _build_generator(resolution=resolution)

    target = torch.rand(1, 3, resolution, resolution)
    mask = torch.ones(1, 1, resolution, resolution)

    params = list(encoder.parameters()) + list(inter.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=5, total_steps=100)

    losses = []
    for step in range(100):
        optimizer.zero_grad()
        rgb, _mask_out = decoder(inter(encoder(target)))
        loss = masked_reconstruction_loss(rgb, target, mask)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.2, f"loss did not converge: {losses[0]} -> {losses[-1]}"
    assert losses[-1] < losses[len(losses) // 2], "loss did not keep improving in the second half"

    with torch.no_grad():
        final_rgb, _ = decoder(inter(encoder(target)))
    final_pixel_error = (final_rgb - target).abs().mean().item()
    assert final_pixel_error < 0.1, f"final reconstruction not close to target: {final_pixel_error}"
