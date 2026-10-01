from __future__ import annotations

import hashlib
import math
import warnings as py_warnings

import torch

from breast_pretrain.train.masked_latent_smoke import (
    build_patch_mask_sampling_priors,
    generate_patch_mask,
    validate_gaze_loss_mode,
    validate_mask_strategy,
)
from breast_pretrain.train.reproducibility import build_patch_mask_sequence
from breast_pretrain.train.stage1_joint.types import (
    MaskingRuntimeState,
    MaskingStepOutput,
    Stage1JointBatch,
    Stage1JointTrainerConfig,
)
from breast_pretrain.train.tissue_aware_gaze import build_stage1_patch_gaze_weights


def _coverage_summary(
    attention_tokens: torch.Tensor,
    patch_mask: torch.Tensor,
    valid_content_patch_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mass = attention_tokens.detach().clamp_min(0.0)
    if valid_content_patch_mask is not None:
        mass = mass * valid_content_patch_mask.to(device=mass.device, dtype=mass.dtype)
    total = mass.sum(dim=1).clamp_min(1e-6)
    masked = (mass * patch_mask.to(dtype=mass.dtype)).sum(dim=1) / total
    visible = (mass * (~patch_mask).to(dtype=mass.dtype)).sum(dim=1) / total
    return masked, visible


def _mask_composition_summary(
    high_conf_tokens: torch.Tensor,
    patch_mask: torch.Tensor,
    eps: float,
    valid_content_patch_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = patch_mask.to(dtype=torch.bool)
    if valid_content_patch_mask is not None:
        content = valid_content_patch_mask.to(device=masked.device, dtype=torch.bool)
        masked = masked & content
    masked_count = masked.sum(dim=1).clamp_min(1).to(dtype=torch.float32)
    high_conf = high_conf_tokens.detach() > float(eps)
    masked_high_conf = (masked & high_conf).sum(dim=1).to(dtype=torch.float32)
    high_conf_quota_actual = masked_high_conf / masked_count
    random_mask_fraction_actual = 1.0 - high_conf_quota_actual
    return high_conf_quota_actual, random_mask_fraction_actual


def _positive_prior(tokens: torch.Tensor, eps: float) -> bool:
    return bool(tokens.detach().clamp_min(0.0).sum().item() > float(eps))


def _deterministic_repair_order(
    indices: torch.Tensor,
    *,
    image_id: str,
    seed: int,
    step_index: int,
    purpose: str,
) -> torch.Tensor:
    """Return a stable, non-spatial ordering without consuming sampler RNG state."""
    ranked = []
    for token_index in indices.detach().cpu().tolist():
        payload = f"{seed}|{step_index}|{image_id}|{purpose}|{token_index}".encode("utf-8")
        rank = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big")
        ranked.append((rank, int(token_index)))
    ranked.sort()
    return torch.tensor([item[1] for item in ranked], dtype=torch.long, device=indices.device)


def _apply_visible_salient_floor(
    *,
    patch_mask: torch.Tensor,
    high_conf_tokens: torch.Tensor,
    valid_content_patch_mask: torch.Tensor | None,
    image_ids: list[str],
    configured_floor: float,
    gaze_mask_eps: float,
    seed: int,
    step_index: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
]:
    """Enforce the formal visible-salient floor after existing mask sampling.

    Membership intentionally reuses the existing high-conf sampler rule
    (high_conf_scores > gaze_mask_eps), rather than introducing a new threshold.
    """
    repaired_mask = patch_mask.to(dtype=torch.bool).clone()
    if valid_content_patch_mask is None:
        valid_mask = torch.ones_like(repaired_mask, dtype=torch.bool)
    else:
        valid_mask = valid_content_patch_mask.to(device=repaired_mask.device, dtype=torch.bool)
    high_conf_membership = high_conf_tokens.to(device=repaired_mask.device) > float(gaze_mask_eps)
    salient_mask = high_conf_membership & valid_mask
    warnings_list: list[str] = []

    for index, image_id in enumerate(image_ids):
        salient = salient_mask[index]
        valid = valid_mask[index]
        total_salient = int(salient.sum().item())
        if total_salient <= 0 or configured_floor <= 0.0:
            continue

        current_mask = repaired_mask[index]
        current_visible = int((salient & ~current_mask).sum().item())
        required_visible = int(math.ceil(float(configured_floor) * total_salient))
        if current_visible >= required_visible:
            continue

        valid_count = int(valid.sum().item())
        mask_count = int((current_mask & valid).sum().item())
        if mask_count > valid_count - required_visible:
            raise RuntimeError(
                "formal visible-salient floor infeasible: "
                f"image_id={image_id}; valid_content_count={valid_count}; mask_count={mask_count}; "
                f"total_salient_count={total_salient}; "
                f"required_visible_salient_count={required_visible}; "
                f"current_visible_salient_count={current_visible}; "
                f"configured_floor={configured_floor:.6f}"
            )

        repair_count = required_visible - current_visible
        masked_salient = torch.nonzero(salient & current_mask, as_tuple=False).flatten()
        replacements = torch.nonzero(valid & ~salient & ~current_mask, as_tuple=False).flatten()
        if int(masked_salient.numel()) < repair_count or int(replacements.numel()) < repair_count:
            raise RuntimeError(
                "formal visible-salient floor repair candidates are insufficient: "
                f"image_id={image_id}; valid_content_count={valid_count}; mask_count={mask_count}; "
                f"total_salient_count={total_salient}; "
                f"required_visible_salient_count={required_visible}; "
                f"current_visible_salient_count={current_visible}; "
                f"configured_floor={configured_floor:.6f}"
            )
        release = _deterministic_repair_order(
            masked_salient,
            image_id=image_id,
            seed=seed,
            step_index=step_index,
            purpose="release_masked_salient",
        )[:repair_count]
        replacement = _deterministic_repair_order(
            replacements,
            image_id=image_id,
            seed=seed,
            step_index=step_index,
            purpose="mask_visible_non_salient",
        )[:repair_count]
        repaired_mask[index, release] = False
        repaired_mask[index, replacement] = True
        warnings_list.append(
            "visible_salient_floor_repaired:"
            f"image_id={image_id}; released={repair_count}; floor={configured_floor:.6f}"
        )

    masked_salient_count = (salient_mask & repaired_mask).sum(dim=1).to(dtype=torch.long)
    total_salient_count = salient_mask.sum(dim=1).to(dtype=torch.long)
    visible_salient_count = total_salient_count - masked_salient_count
    q_vis = torch.where(
        total_salient_count > 0,
        visible_salient_count.to(dtype=torch.float32) / total_salient_count.to(dtype=torch.float32),
        torch.zeros_like(total_salient_count, dtype=torch.float32),
    )
    violation_count = (
        (total_salient_count > 0)
        & (visible_salient_count < torch.ceil(total_salient_count.to(dtype=torch.float32) * configured_floor).to(dtype=torch.long))
    ).to(dtype=torch.long)
    if int(violation_count.sum().item()) > 0:
        raise RuntimeError("formal visible-salient floor violation remained after deterministic repair.")
    return (
        repaired_mask,
        total_salient_count,
        masked_salient_count,
        visible_salient_count,
        q_vis,
        violation_count,
        warnings_list,
    )


def _is_weak_qc(
    prior_status: str,
    audit_status: str,
    coverage_ratio: float,
    high_conf_area_ratio: float,
    inside_ratio: float,
) -> bool:
    text = f"{prior_status} {audit_status}".lower()
    if any(term in text for term in ("weak", "review", "warning", "pass_with_warnings")):
        return True
    if 0.0 < float(coverage_ratio) < 0.05:
        return True
    if 0.0 < float(high_conf_area_ratio) < 0.005:
        return True
    if 0.0 < float(inside_ratio) < 0.25:
        return True
    return False


def _build_adaptive_gaze_mask(
    config: Stage1JointTrainerConfig,
    batch: Stage1JointBatch,
    device: torch.device,
    num_patches: int,
    patch_gaze_scores: torch.Tensor,
    high_conf_patch_scores: torch.Tensor,
    generator: torch.Generator | None,
    valid_content_patch_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor, torch.Tensor, tuple[str, ...], list[str]]:
    masks: list[torch.Tensor] = []
    policies: list[str] = []
    quotas: list[float] = []
    random_fractions: list[float] = []
    fallback_reasons: list[str] = []
    warnings_list: list[str] = []
    valid_sources = {"observed_gaze", "diffeye_generated_gaze"}
    strong_statuses = {"usable_prior", "pass", "accepted", "usable", "confirmed"}
    batch_size = int(batch.image.shape[0])

    for index in range(batch_size):
        source = str(batch.gaze_supervision_sources[index]).strip().lower()
        modality = str(batch.modalities[index]).strip().lower()
        prior_status = str(batch.prior_statuses[index]).strip().lower()
        audit_status = str(batch.audit_statuses[index]).strip().lower()
        has_prior = _positive_prior(patch_gaze_scores[index], config.masking.gaze_mask_eps)
        is_gaze_source = source in valid_sources
        strong_qc = (not prior_status or prior_status in strong_statuses) and (
            not audit_status or audit_status in strong_statuses
        )

        fallback_reason = ""
        policy = "adaptive_gaze_guided"
        quota = float(config.masking.high_conf_mask_quota)
        random_fraction = float(config.masking.min_random_mask_fraction)
        strategy = "gaze_biased"
        alpha = float(config.masking.gaze_mask_sampling_alpha)

        if not is_gaze_source:
            fallback_reason = f"gaze_source_not_valid:{source or 'missing'}"
        elif not has_prior:
            fallback_reason = "gaze_prior_missing_or_zero"
        elif _is_weak_qc(
            prior_status=prior_status,
            audit_status=audit_status,
            coverage_ratio=batch.coverage_ratios[index],
            high_conf_area_ratio=batch.high_conf_area_ratios[index],
            inside_ratio=batch.inside_ratios[index],
        ) or not strong_qc:
            policy = "adaptive_weak_qc"
            quota = max(0.05, quota * 0.5)
            alpha = max(0.05, alpha * 0.5)
            random_fraction = min(0.9, max(random_fraction, 1.0 - quota))

        if fallback_reason:
            normalized_gaze_loss = validate_gaze_loss_mode(config.masking.gaze_loss_mode)
            if normalized_gaze_loss == "soft_attention_plus_high_conf":
                high_conf_for_sample = high_conf_patch_scores[index:index + 1]
                if valid_content_patch_mask is not None:
                    valid_row = valid_content_patch_mask[index:index + 1].to(
                        device=device, dtype=torch.bool,
                    )
                    high_conf_for_sample = high_conf_for_sample.to(device=device) * valid_row.to(
                        dtype=high_conf_for_sample.dtype,
                    )
                if _positive_prior(high_conf_for_sample, config.masking.gaze_mask_eps):
                    fallback_reason = ""
                    policy = "high_conf_guided_from_sidecar"
                    strategy = "high_conf_quota"
                    alpha = float(config.masking.gaze_mask_sampling_alpha)

        if fallback_reason:
            policy = "random_fallback"
            quota = 0.0
            random_fraction = 1.0
            strategy = "random"
            warnings_list.append(f"adaptive_gaze_masking_fallback:{batch.image_ids[index]}:{fallback_reason}")

        valid_indices = None
        local_num_patches = num_patches
        local_gaze = patch_gaze_scores[index:index + 1]
        local_high_conf = high_conf_patch_scores[index:index + 1]
        if valid_content_patch_mask is not None:
            valid_row = valid_content_patch_mask[index].to(device=device, dtype=torch.bool)
            valid_indices = torch.nonzero(valid_row, as_tuple=False).flatten()
            if int(valid_indices.numel()) <= 0:
                raise ValueError(f"{batch.image_ids[index]}: no valid-content patches available for masking.")
            local_num_patches = int(valid_indices.numel())
            local_gaze = local_gaze[:, valid_indices]
            local_high_conf = local_high_conf[:, valid_indices]

        with py_warnings.catch_warnings(record=True) as caught_warnings:
            py_warnings.simplefilter("always")
            local_mask = generate_patch_mask(
                batch_size=1,
                num_patches=local_num_patches,
                mask_ratio=config.masking.mask_ratio,
                device=device,
                generator=generator,
                mask_strategy=strategy,
                patch_gaze_scores=local_gaze,
                high_conf_mask=local_high_conf,
                gaze_mask_sampling_alpha=alpha,
                high_conf_mask_quota=quota,
                min_random_mask_fraction=random_fraction,
                mask_sampling_temperature=config.masking.mask_sampling_temperature,
                gaze_mask_eps=config.masking.gaze_mask_eps,
            )
        if valid_indices is None:
            mask = local_mask
        else:
            mask = torch.zeros((1, num_patches), dtype=torch.bool, device=device)
            mask[0, valid_indices] = local_mask[0]
        warnings_list.extend(str(item.message) for item in caught_warnings)
        masks.append(mask[0])
        policies.append(policy)
        quotas.append(quota)
        random_fractions.append(random_fraction)
        fallback_reasons.append(fallback_reason)

    return (
        torch.stack(masks, dim=0).to(device=device),
        tuple(policies),
        torch.tensor(quotas, dtype=torch.float32, device=device),
        torch.tensor(random_fractions, dtype=torch.float32, device=device),
        tuple(fallback_reasons),
        warnings_list,
    )


def _generate_content_candidate_mask(
    *,
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device,
    generator: torch.Generator | None,
    mask_strategy: str,
    content_mask: torch.Tensor,
    fixed_mask: torch.Tensor | None = None,
    patch_gaze_scores: torch.Tensor | None = None,
    high_conf_patch_scores: torch.Tensor | None = None,
    gaze_mask_sampling_alpha: float = 0.7,
    high_conf_mask_quota: float = 0.6,
    min_random_mask_fraction: float = 0.3,
    mask_sampling_temperature: float = 1.0,
    gaze_mask_eps: float = 1e-6,
) -> torch.Tensor:
    masks: list[torch.Tensor] = []
    for index in range(batch_size):
        valid_indices = torch.nonzero(content_mask[index].to(dtype=torch.bool), as_tuple=False).flatten()
        if int(valid_indices.numel()) <= 0:
            raise ValueError(f"sample {index}: no valid-content patches available for masking.")
        local_fixed = fixed_mask[index:index + 1, valid_indices] if fixed_mask is not None else None
        local_gaze = patch_gaze_scores[index:index + 1, valid_indices] if patch_gaze_scores is not None else None
        local_high_conf = (
            high_conf_patch_scores[index:index + 1, valid_indices]
            if high_conf_patch_scores is not None
            else None
        )
        local_mask = generate_patch_mask(
            batch_size=1,
            num_patches=int(valid_indices.numel()),
            mask_ratio=mask_ratio,
            device=device,
            generator=generator,
            fixed_mask=local_fixed,
            mask_strategy=mask_strategy,
            patch_gaze_scores=local_gaze,
            high_conf_mask=local_high_conf,
            gaze_mask_sampling_alpha=gaze_mask_sampling_alpha,
            high_conf_mask_quota=high_conf_mask_quota,
            min_random_mask_fraction=min_random_mask_fraction,
            mask_sampling_temperature=mask_sampling_temperature,
            gaze_mask_eps=gaze_mask_eps,
        )
        full_mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
        full_mask[valid_indices] = local_mask[0]
        masks.append(full_mask)
    return torch.stack(masks, dim=0)


def build_masking_runtime_state(
    config: Stage1JointTrainerConfig,
    num_patches: int,
) -> MaskingRuntimeState:
    normalized_gaze_loss_mode = validate_gaze_loss_mode(config.masking.gaze_loss_mode)
    configured_mask_strategy = validate_mask_strategy(config.masking.mask_strategy)
    normalized_mask_strategy = (
        "random" if normalized_gaze_loss_mode == "no_gaze" else configured_mask_strategy
    )
    mask_prior_mode = "observed_gaze"
    if normalized_gaze_loss_mode == "no_gaze":
        mask_prior_mode = "disabled"
    elif normalized_gaze_loss_mode == "center_prior":
        mask_prior_mode = "center_prior"
    elif normalized_gaze_loss_mode == "shuffled_gaze":
        mask_prior_mode = "shuffled_gaze"

    if (
        config.reproducibility.reuse_patch_mask
        and normalized_mask_strategy == "random"
        and int(num_patches) > 0
    ):
        shared_patch_masks, mask_checksum = build_patch_mask_sequence(
            batch_size=config.data.batch_size,
            num_patches=num_patches,
            mask_ratio=config.masking.mask_ratio,
            max_steps=config.train.max_steps,
            seed=config.reproducibility.seed,
        )
    else:
        shared_patch_masks = None
        mask_checksum = "generated_per_step"

    mask_generator = None
    if shared_patch_masks is None and normalized_mask_strategy != "random":
        mask_generator = torch.Generator(device="cpu")
        mask_generator.manual_seed(int(config.reproducibility.seed))

    return MaskingRuntimeState(
        normalized_gaze_loss_mode=normalized_gaze_loss_mode,
        configured_mask_strategy=configured_mask_strategy,
        normalized_mask_strategy=normalized_mask_strategy,
        mask_prior_mode=mask_prior_mode,
        mask_checksum=mask_checksum,
        shared_patch_masks=shared_patch_masks,
        mask_generator=mask_generator,
    )


def build_masking_step_output(
    config: Stage1JointTrainerConfig,
    batch: Stage1JointBatch,
    device: torch.device,
    num_patches: int,
    step_index: int,
    runtime_state: MaskingRuntimeState,
    *,
    patch_gaze_scores: torch.Tensor | None = None,
    high_conf_patch_scores: torch.Tensor | None = None,
    valid_content_patch_mask: torch.Tensor | None = None,
) -> MaskingStepOutput:
    content_mask = None
    if valid_content_patch_mask is not None:
        content_mask = valid_content_patch_mask.to(device=device, dtype=torch.bool)
        if content_mask.ndim == 1:
            content_mask = content_mask.unsqueeze(0).expand(int(batch.image.shape[0]), -1)
        if tuple(content_mask.shape) != (int(batch.image.shape[0]), int(num_patches)):
            raise ValueError(
                "valid_content_patch_mask must have shape [B,N]; "
                f"got {tuple(content_mask.shape)}, expected "
                f"({int(batch.image.shape[0])},{int(num_patches)})"
            )
        if patch_gaze_scores is not None:
            patch_gaze_scores = patch_gaze_scores.to(device=device) * content_mask.to(dtype=patch_gaze_scores.dtype)
        if high_conf_patch_scores is not None:
            high_conf_patch_scores = high_conf_patch_scores.to(device=device) * content_mask.to(dtype=high_conf_patch_scores.dtype)
    # Fail-fast: if pre-computed priors are 196-token (from legacy 224x224 path), block
    if patch_gaze_scores is not None and patch_gaze_scores.shape[1] == 196 and num_patches != 196:
        raise RuntimeError(
            f"FATAL: patch_gaze_scores has 196 tokens but resolved num_patches={num_patches}. "
            f"Training must use dynamic grid priors, not legacy 224x224 -> patch_size=16 -> 196 path."
        )
    if high_conf_patch_scores is not None and high_conf_patch_scores.shape[1] == 196 and num_patches != 196:
        raise RuntimeError(
            f"FATAL: high_conf_patch_scores has 196 tokens but resolved num_patches={num_patches}. "
            f"Training must use materialized high-conf priors, not legacy 224x224 -> 196 path."
        )
    # Fail-fast: if computed priors from 224x224 would produce 196 tokens for non-196 grid
    if num_patches != 196 and runtime_state.normalized_mask_strategy not in ("random",):
        # Verify that we're NOT using the legacy path which produces 196 tokens
        legacy_num_patches = (batch.attention_map.shape[-2] // config.model.patch_size) * (
            batch.attention_map.shape[-1] // config.model.patch_size
        )
        if legacy_num_patches == 196 and patch_gaze_scores is None:
            raise RuntimeError(
                f"FATAL: Would compute priors at legacy 196-token grid from 224x224 attention, "
                f"but resolved num_patches={num_patches}. "
                f"pre-computed dynamic priors must be passed to build_masking_step_output."
            )
    if runtime_state.shared_patch_masks is not None:
        fixed_mask = runtime_state.shared_patch_masks[step_index - 1]
        if content_mask is not None:
            patch_mask = _generate_content_candidate_mask(
                batch_size=int(batch.image.shape[0]),
                num_patches=num_patches,
                mask_ratio=config.masking.mask_ratio,
                device=device,
                generator=None,
                fixed_mask=fixed_mask[: int(batch.image.shape[0])],
                mask_strategy="random",
                content_mask=content_mask,
            )
        else:
            patch_mask = generate_patch_mask(
                batch_size=int(batch.image.shape[0]),
                num_patches=num_patches,
                mask_ratio=config.masking.mask_ratio,
                device=device,
                fixed_mask=fixed_mask[: int(batch.image.shape[0])],
            )
        warnings_list: list[str] = []
    elif runtime_state.normalized_mask_strategy == "random":
        if content_mask is not None:
            patch_mask = _generate_content_candidate_mask(
                batch_size=int(batch.image.shape[0]),
                num_patches=num_patches,
                mask_ratio=config.masking.mask_ratio,
                device=device,
                generator=None,
                mask_strategy="random",
                content_mask=content_mask,
            )
        else:
            patch_mask = generate_patch_mask(
                batch_size=int(batch.image.shape[0]),
                num_patches=num_patches,
                mask_ratio=config.masking.mask_ratio,
                device=device,
            )
        warnings_list = []
    else:
        if patch_gaze_scores is None or high_conf_patch_scores is None:
            patch_gaze_scores, high_conf_patch_scores = build_patch_mask_sampling_priors(
                attention_map=batch.attention_map,
                high_conf_mask=batch.high_conf_mask,
                patch_size=config.model.patch_size,
                prior_mode=runtime_state.mask_prior_mode,
                random_seed=config.reproducibility.seed + step_index - 1,
            )
        if runtime_state.normalized_mask_strategy == "adaptive_gaze_masking":
            (
                patch_mask,
                mask_policy_used,
                adaptive_gaze_quota,
                adaptive_random_fraction,
                fallback_reason,
                warnings_list,
            ) = _build_adaptive_gaze_mask(
                config=config,
                batch=batch,
                device=device,
                num_patches=num_patches,
                patch_gaze_scores=patch_gaze_scores,
                high_conf_patch_scores=high_conf_patch_scores,
                generator=runtime_state.mask_generator,
                valid_content_patch_mask=content_mask,
            )
        else:
            if runtime_state.normalized_mask_strategy == "tissue_aware":
                tissue_patch_weights, _ = build_stage1_patch_gaze_weights(
                    image=batch.image,
                    attention_map=batch.attention_map,
                    high_conf_mask=batch.high_conf_mask,
                    patch_size=config.model.patch_size,
                    gaze_loss_mode=config.masking.gaze_loss_mode,
                    gaze_weight_alpha=config.masking.gaze_weight_alpha,
                    random_seed=config.reproducibility.seed + step_index - 1,
                )
                patch_gaze_scores = tissue_patch_weights.detach().cpu() - 1.0
            with py_warnings.catch_warnings(record=True) as caught_warnings:
                py_warnings.simplefilter("always")
                if content_mask is not None:
                    patch_mask = _generate_content_candidate_mask(
                        batch_size=int(batch.image.shape[0]),
                        num_patches=num_patches,
                        mask_ratio=config.masking.mask_ratio,
                        device=device,
                        generator=runtime_state.mask_generator,
                        mask_strategy=runtime_state.normalized_mask_strategy,
                        content_mask=content_mask,
                        patch_gaze_scores=patch_gaze_scores,
                        high_conf_patch_scores=high_conf_patch_scores,
                        gaze_mask_sampling_alpha=config.masking.gaze_mask_sampling_alpha,
                        high_conf_mask_quota=config.masking.high_conf_mask_quota,
                        min_random_mask_fraction=config.masking.min_random_mask_fraction,
                        mask_sampling_temperature=config.masking.mask_sampling_temperature,
                        gaze_mask_eps=config.masking.gaze_mask_eps,
                    )
                else:
                    patch_mask = generate_patch_mask(
                        batch_size=int(batch.image.shape[0]),
                        num_patches=num_patches,
                        mask_ratio=config.masking.mask_ratio,
                        device=device,
                        generator=runtime_state.mask_generator,
                        mask_strategy=runtime_state.normalized_mask_strategy,
                        patch_gaze_scores=patch_gaze_scores,
                        high_conf_mask=high_conf_patch_scores,
                        gaze_mask_sampling_alpha=config.masking.gaze_mask_sampling_alpha,
                        high_conf_mask_quota=config.masking.high_conf_mask_quota,
                        min_random_mask_fraction=config.masking.min_random_mask_fraction,
                        mask_sampling_temperature=config.masking.mask_sampling_temperature,
                        gaze_mask_eps=config.masking.gaze_mask_eps,
                    )
            warnings_list = [str(item.message) for item in caught_warnings]

    if patch_gaze_scores is not None and patch_gaze_scores.shape[1] != 196:
        # Use pre-computed dynamic priors at final grid
        patch_weights = patch_gaze_scores.detach().clone()
        attention_tokens = patch_gaze_scores.detach().clone()
        high_conf_tokens = (
            high_conf_patch_scores.detach().clone()
            if high_conf_patch_scores is not None
            else torch.zeros_like(patch_gaze_scores)
        )
    else:
        patch_weights, _summary = build_stage1_patch_gaze_weights(
            image=batch.image,
            attention_map=batch.attention_map,
            high_conf_mask=batch.high_conf_mask,
            patch_size=config.model.patch_size,
            gaze_loss_mode=runtime_state.normalized_gaze_loss_mode,
            gaze_weight_alpha=config.masking.gaze_weight_alpha,
            random_seed=config.reproducibility.seed + step_index - 1,
        )
        attention_tokens, high_conf_tokens = build_patch_mask_sampling_priors(
            attention_map=batch.attention_map,
            high_conf_mask=batch.high_conf_mask,
            patch_size=config.model.patch_size,
        )
    if content_mask is not None:
        patch_weights = patch_weights.to(device=device) * content_mask.to(dtype=patch_weights.dtype)
        attention_tokens = attention_tokens.to(device=device) * content_mask.to(dtype=attention_tokens.dtype)
        high_conf_tokens = high_conf_tokens.to(device=device) * content_mask.to(dtype=high_conf_tokens.dtype)
    (
        patch_mask,
        total_salient_count,
        masked_salient_count,
        visible_salient_count,
        q_vis,
        visible_salient_floor_violation_count,
        floor_warnings,
    ) = _apply_visible_salient_floor(
        patch_mask=patch_mask,
        high_conf_tokens=high_conf_tokens,
        valid_content_patch_mask=content_mask,
        image_ids=batch.image_ids,
        configured_floor=float(config.masking.min_visible_salient_fraction),
        gaze_mask_eps=config.masking.gaze_mask_eps,
        seed=config.reproducibility.seed,
        step_index=step_index,
    )
    warnings_list.extend(floor_warnings)
    if "mask_policy_used" not in locals():
        batch_size = int(batch.image.shape[0])
        mask_policy_used = tuple([runtime_state.normalized_mask_strategy] * batch_size)
        adaptive_gaze_quota = torch.zeros(batch_size, dtype=torch.float32, device=device)
        adaptive_random_fraction = torch.ones(batch_size, dtype=torch.float32, device=device)
        fallback_reason = tuple([""] * batch_size)
    masked_gaze_coverage, visible_gaze_coverage = _coverage_summary(
        attention_tokens,
        patch_mask,
        valid_content_patch_mask=content_mask,
    )
    high_conf_mask_quota_actual, random_mask_fraction_actual = _mask_composition_summary(
        high_conf_tokens=high_conf_tokens,
        patch_mask=patch_mask,
        eps=config.masking.gaze_mask_eps,
        valid_content_patch_mask=content_mask,
    )

    # Per-modality mask policy statistics
    adaptive_gaze_guided_by_modality: dict[str, int] = {}
    adaptive_weak_qc_by_modality: dict[str, int] = {}
    random_fallback_by_modality: dict[str, int] = {}
    fallback_reason_histogram: dict[str, int] = {}

    batch_modalities = batch.modalities
    for idx, mod in enumerate(batch_modalities):
        mod = mod.strip().lower()
        policy = mask_policy_used[idx] if idx < len(mask_policy_used) else "unknown"
        reason = fallback_reason[idx] if idx < len(fallback_reason) else ""

        if policy == "adaptive_gaze_guided":
            adaptive_gaze_guided_by_modality[mod] = adaptive_gaze_guided_by_modality.get(mod, 0) + 1
        elif policy == "adaptive_weak_qc":
            adaptive_weak_qc_by_modality[mod] = adaptive_weak_qc_by_modality.get(mod, 0) + 1
        elif policy == "random_fallback":
            random_fallback_by_modality[mod] = random_fallback_by_modality.get(mod, 0) + 1

        if reason:
            fallback_reason_histogram[reason] = fallback_reason_histogram.get(reason, 0) + 1

    return MaskingStepOutput(
        patch_mask=patch_mask,
        patch_weights=patch_weights,
        attention_tokens=attention_tokens,
        high_conf_tokens=high_conf_tokens,
        mask_policy_used=mask_policy_used,
        adaptive_gaze_quota=adaptive_gaze_quota,
        adaptive_random_fraction=adaptive_random_fraction,
        fallback_reason=fallback_reason,
        masked_gaze_coverage=masked_gaze_coverage,
        visible_gaze_coverage=visible_gaze_coverage,
        high_conf_mask_quota_actual=high_conf_mask_quota_actual,
        random_mask_fraction_actual=random_mask_fraction_actual,
        warnings=warnings_list,
        total_salient_count=total_salient_count,
        masked_salient_count=masked_salient_count,
        visible_salient_count=visible_salient_count,
        q_vis=q_vis,
        visible_salient_floor_violation_count=visible_salient_floor_violation_count,
        adaptive_gaze_guided_by_modality=adaptive_gaze_guided_by_modality,
        adaptive_weak_qc_by_modality=adaptive_weak_qc_by_modality,
        random_fallback_by_modality=random_fallback_by_modality,
        fallback_reason_histogram=fallback_reason_histogram,
    )
