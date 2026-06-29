import torch

from verl.trainer.ppo.core_algos import (
    compute_self_distillation_reweighted_advantages,
    is_self_distillation_loss_mode,
    is_self_distillation_reweight_loss_mode,
)


def test_self_distillation_reweight_loss_mode_aliases():
    assert is_self_distillation_loss_mode("sdpo")
    assert is_self_distillation_loss_mode("srpo")
    assert is_self_distillation_loss_mode("rlsd")
    assert is_self_distillation_reweight_loss_mode("rlsd")
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
