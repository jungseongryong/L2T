import torch
import pytest

from verl.trainer.ppo.core_algos import (
    compute_evolving_teacher_policy_loss,
    compute_srpo_policy_loss,
    compute_self_distillation_reweighted_advantages,
    is_self_distillation_loss_mode,
    is_self_distillation_reweight_loss_mode,
)
from verl.workers.config.actor import EvolvingTeacherConfig, SelfDistillationConfig


class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]


def test_self_distillation_reweight_loss_mode_aliases():
    assert is_self_distillation_loss_mode("sdpo")
    assert is_self_distillation_loss_mode("srpo")
    assert is_self_distillation_loss_mode("rlsd")
    assert is_self_distillation_loss_mode("rlrt")
    assert is_self_distillation_reweight_loss_mode("rlsd")
    assert is_self_distillation_reweight_loss_mode("rlrt")
    assert not is_self_distillation_reweight_loss_mode("sdpo")
    assert not is_self_distillation_reweight_loss_mode("srpo")


def test_rlsd_reweights_with_teacher_over_student_ratio():
    student_log_probs = torch.tensor([[-2.0, -2.0]])
    teacher_log_probs = torch.tensor([[-1.0, -3.0]])
    advantages = torch.tensor([[2.0, -2.0]])
    response_mask = torch.ones_like(advantages)
    context_mask = torch.tensor([1.0])
    cfg = {"token_reweight_lambda": 0.5, "token_reweight_eps_w": 0.2}

    refined, metrics = compute_self_distillation_reweighted_advantages(
        loss_mode="rlsd",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        self_distillation_config=cfg,
        self_distillation_mask=context_mask,
    )

    clipped_weight = 1.2
    expected_modulator = (1.0 - 0.5) + 0.5 * clipped_weight
    assert torch.allclose(refined, advantages * expected_modulator)
    assert metrics["self_distillation/token_reweight_w_clip_frac"] == 1.0


def test_token_reweight_lambda_linear_decay():
    student_log_probs = torch.tensor([[-1.0]])
    teacher_log_probs = torch.tensor([[0.0]])
    advantages = torch.ones(1, 1)
    response_mask = torch.ones_like(advantages)
    context_mask = torch.tensor([1.0])
    cfg = {
        "token_reweight_lambda": 1.0,
        "token_reweight_eps_w": 10.0,
        "token_reweight_decay_steps": 10,
    }

    refined, metrics = compute_self_distillation_reweighted_advantages(
        loss_mode="rlsd",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        self_distillation_config=cfg,
        self_distillation_mask=context_mask,
        global_step=6,
    )

    expected_lambda = 0.5
    expected = (1.0 - expected_lambda) + expected_lambda * torch.exp(torch.tensor(1.0)).item()
    assert torch.allclose(refined, torch.tensor([[expected]]))
    assert metrics["self_distillation/token_reweight_lambda"] == expected_lambda


def test_rlrt_reverses_teacher_signal_and_gates_to_correct_rollouts():
    student_log_probs = torch.tensor([[-1.0, -3.0], [-1.0, -3.0]])
    teacher_log_probs = torch.tensor([[-2.0, -2.0], [-2.0, -2.0]])
    advantages = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    response_mask = torch.ones_like(advantages)
    context_mask = torch.tensor([1.0, 1.0])
    correct_mask = torch.tensor([1.0, 0.0])
    cfg = {"token_reweight_lambda": 1.0, "token_reweight_eps_w": 10.0}

    refined, metrics = compute_self_distillation_reweighted_advantages(
        loss_mode="rlrt",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        self_distillation_config=cfg,
        self_distillation_mask=context_mask,
        self_distillation_correct_mask=correct_mask,
    )

    # RLRT uses exp(sign(A) * (log P_S - log P_T)) and applies it only to correct samples.
    expected = torch.tensor(
        [
            [2.0 * torch.exp(torch.tensor(1.0)).item(), 2.0 * torch.exp(torch.tensor(-1.0)).item()],
            [2.0, 2.0],
        ]
    )
    assert torch.allclose(refined, expected)
    assert metrics["self_distillation/token_reweight_is_rlrt"] == 1.0
    assert metrics["self_distillation/token_reweight_reward_gate_fraction"] == 0.5


def test_srpo_uses_single_token_denominator_across_routed_branches():
    old_log_prob = torch.tensor([[0.0], [0.0], [3.0]])
    log_prob = torch.tensor([[0.0], [0.0], [3.0]])
    teacher_log_probs = torch.tensor([[0.0], [0.0], [2.0]])
    advantages = torch.tensor([[2.0], [4.0], [0.0]])
    response_mask = torch.ones_like(advantages)
    self_distillation_mask = torch.tensor([0.0, 0.0, 1.0])
    self_distillation_correct_mask = torch.tensor([1.0, 1.0, 0.0])
    actor_cfg = AttrDict(
        clip_ratio=0.2,
        clip_ratio_low=None,
        clip_ratio_high=None,
        clip_ratio_c=3.0,
    )
    sd_cfg = AttrDict(
        full_logit_distillation=False,
        alpha=1.0,
        is_clip=None,
        srpo_dynamic_weighting=False,
    )

    loss, metrics = compute_srpo_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        teacher_log_probs=teacher_log_probs,
        self_distillation_config=sd_cfg,
        self_distillation_mask=self_distillation_mask,
        self_distillation_correct_mask=self_distillation_correct_mask,
        config=actor_cfg,
    )

    # GRPO token losses: -2 and -4. SDPO token loss: (3 - 2).detach() * 3 = 3.
    # The SRPO paper averages over all routed tokens once: (-2 - 4 + 3) / 3 = -1.
    assert torch.allclose(loss, torch.tensor(-1.0))
    assert metrics["srpo/grpo_loss"] == -3.0
    assert metrics["srpo/sdpo_loss"] == 3.0


def test_evolving_teacher_policy_loss_uses_ratio_one_teacher_view_pg():
    teacher_log_probs = torch.zeros(1, 2, requires_grad=True)
    advantages = torch.tensor([[2.0, -3.0]])
    response_mask = torch.ones_like(advantages)
    cfg = AttrDict(enable=True, loss_weight=0.1, mask="all")

    loss, metrics = compute_evolving_teacher_policy_loss(
        teacher_log_probs=teacher_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        evolving_teacher_config=cfg,
    )
    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.0))
    assert teacher_log_probs.grad[0, 0] < 0
    assert teacher_log_probs.grad[0, 1] > 0
    assert metrics["et/active_token_fraction"] == 1.0
    assert metrics["et/active_sample_fraction"] == 1.0


def test_evolving_teacher_policy_loss_can_mask_incorrect_context_samples():
    teacher_log_probs = torch.zeros(3, 1, requires_grad=True)
    advantages = torch.ones(3, 1)
    response_mask = torch.ones_like(advantages)
    context_mask = torch.tensor([1.0, 1.0, 0.0])
    correct_mask = torch.tensor([1.0, 0.0, 0.0])
    cfg = AttrDict(enable=True, loss_weight=0.1, mask="incorrect_context")

    loss, metrics = compute_evolving_teacher_policy_loss(
        teacher_log_probs=teacher_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        evolving_teacher_config=cfg,
        self_distillation_mask=context_mask,
        self_distillation_correct_mask=correct_mask,
    )
    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.0))
    assert teacher_log_probs.grad[0, 0] == 0.0
    assert teacher_log_probs.grad[1, 0] < 0.0
    assert teacher_log_probs.grad[2, 0] == 0.0
    assert metrics["et/active_token_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["et/active_sample_fraction"] == pytest.approx(1.0 / 3.0)


def test_evolving_teacher_policy_loss_can_filter_negative_advantages():
    teacher_log_probs = torch.zeros(1, 3, requires_grad=True)
    advantages = torch.tensor([[2.0, -3.0, 0.0]])
    response_mask = torch.ones_like(advantages)
    cfg = AttrDict(enable=True, loss_weight=0.1, mask="all", advantage_filter="negative")

    loss, metrics = compute_evolving_teacher_policy_loss(
        teacher_log_probs=teacher_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        evolving_teacher_config=cfg,
    )
    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.0))
    assert teacher_log_probs.grad[0, 0] == 0.0
    assert teacher_log_probs.grad[0, 1] > 0.0
    assert teacher_log_probs.grad[0, 2] == 0.0
    assert metrics["et/active_token_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["et/advantage_filter_negative"] == 1.0


def test_evolving_teacher_policy_loss_can_contrast_teacher_and_student_views():
    teacher_log_probs = torch.zeros(1, 2, requires_grad=True)
    student_log_probs = torch.tensor([[4.0, -4.0]], requires_grad=True)
    advantages = torch.tensor([[2.0, -3.0]])
    response_mask = torch.ones_like(advantages)
    cfg = AttrDict(enable=True, loss_weight=0.1, objective="view_contrast", mask="all")

    loss, metrics = compute_evolving_teacher_policy_loss(
        teacher_log_probs=teacher_log_probs,
        student_log_probs=student_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        evolving_teacher_config=cfg,
    )
    loss.backward()

    assert torch.allclose(loss, torch.tensor(10.0))
    assert teacher_log_probs.grad[0, 0] < 0.0
    assert teacher_log_probs.grad[0, 1] > 0.0
    assert student_log_probs.grad is None
    assert metrics["et/objective_view_contrast"] == 1.0
    assert metrics["et/view_delta_mean"] == pytest.approx(0.0)


def test_evolving_teacher_policy_loss_can_reverse_teacher_view_pg():
    teacher_log_probs = torch.zeros(1, 2, requires_grad=True)
    advantages = torch.tensor([[2.0, -3.0]])
    response_mask = torch.ones_like(advantages)
    cfg = AttrDict(enable=True, loss_weight=0.1, objective="reverse_policy_gradient", mask="all")

    loss, metrics = compute_evolving_teacher_policy_loss(
        teacher_log_probs=teacher_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        evolving_teacher_config=cfg,
    )
    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.0))
    assert teacher_log_probs.grad[0, 0] > 0.0
    assert teacher_log_probs.grad[0, 1] < 0.0
    assert metrics["et/objective_reverse_policy_gradient"] == 1.0


def test_evolving_teacher_policy_loss_can_reverse_view_contrast():
    teacher_log_probs = torch.zeros(1, 2, requires_grad=True)
    student_log_probs = torch.tensor([[4.0, -4.0]], requires_grad=True)
    advantages = torch.tensor([[2.0, -3.0]])
    response_mask = torch.ones_like(advantages)
    cfg = AttrDict(enable=True, loss_weight=0.1, objective="reverse_view_contrast", mask="all")

    loss, metrics = compute_evolving_teacher_policy_loss(
        teacher_log_probs=teacher_log_probs,
        student_log_probs=student_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        evolving_teacher_config=cfg,
    )
    loss.backward()

    assert torch.allclose(loss, torch.tensor(-10.0))
    assert teacher_log_probs.grad[0, 0] > 0.0
    assert teacher_log_probs.grad[0, 1] < 0.0
    assert student_log_probs.grad is None
    assert metrics["et/objective_reverse_view_contrast"] == 1.0
    assert metrics["et/view_delta_mean"] == pytest.approx(0.0)


def test_evolving_teacher_config_requires_no_ema_update():
    with pytest.raises(ValueError, match="EMA teacher updates"):
        SelfDistillationConfig(
            teacher_regularization="ema",
            teacher_update_rate=0.1,
            evolving_teacher=EvolvingTeacherConfig(enable=True, loss_weight=0.1),
        )


def test_self_distillation_config_accepts_evolving_teacher_dict():
    cfg = SelfDistillationConfig(
        teacher_update_rate=0.0,
        evolving_teacher={"enable": True, "loss_weight": 0.1, "mask": "all"},
    )

    assert isinstance(cfg.evolving_teacher, EvolvingTeacherConfig)
    assert cfg.evolving_teacher.enable is True


def test_evolving_teacher_config_accepts_trust_region_teacher():
    cfg = SelfDistillationConfig(
        teacher_regularization="trust-region",
        teacher_update_rate=0.1,
        evolving_teacher=EvolvingTeacherConfig(enable=True, loss_weight=0.1),
    )

    assert cfg.teacher_regularization == "trust-region"
    assert cfg.teacher_update_rate == 0.1
    assert cfg.evolving_teacher.enable is True
