from __future__ import annotations

from pathlib import Path
from typing import Any
from itertools import combinations

import torch
import yaml


GraphNodeBinding = dict[str, Any]


def _normalized_consistency_rules(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize V1 pair rules and V2 axis/group mutual-exclusion rules."""

    legacy_rules = payload.get("consistency_rules", [])
    if isinstance(legacy_rules, list) and legacy_rules:
        return [rule for rule in legacy_rules if isinstance(rule, dict)]

    normalized: list[dict[str, Any]] = []
    for rule in payload.get("rules", []):
        if not isinstance(rule, dict) or rule.get("rule_type") != "mutual_exclusion":
            continue
        node_ids = [str(node_id).strip() for node_id in rule.get("node_ids", []) if str(node_id).strip()]
        for source_node_id, target_node_id in combinations(node_ids, 2):
            normalized.append(
                {
                    "rule_type": "conflicts_with",
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "penalty_weight": float(rule.get("penalty_weight", 1.0)),
                }
            )
    return normalized


class GraphConsistencyLoss:
    """Graph consistency loss for Stage 1 clinical concept logits.

    This is NOT a graph node direct alignment loss. It operates on concept logits
    and applies consistency rules from graph_consistency_rules_v1.yaml as a
    regularization term. Only nodes with observed_mask=1 are used.
    Missing labels are never treated as negative.
    """

    def __init__(self, consistency_rules_path: str | Path):
        self.rules_path = Path(consistency_rules_path)
        self.rules = self._load_rules()
        self._parse_rules()

    def _load_rules(self) -> dict[str, Any]:
        with self.rules_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _parse_rules(self) -> None:
        raw_rules = _normalized_consistency_rules(self.rules)
        self.conflict_pairs: list[tuple[str, str, float]] = []
        self.implication_pairs: list[tuple[str, str, float]] = []
        self.mapping_pairs: list[tuple[str, str, float]] = []
        self.risk_order_pairs: list[tuple[str, str, float]] = []

        for rule in raw_rules:
            rule_type = rule.get("rule_type", "")
            src = rule.get("source_node_id", "")
            tgt = rule.get("target_node_id", "")
            weight = float(rule.get("penalty_weight", 1.0))

            if rule_type == "conflicts_with":
                self.conflict_pairs.append((src, tgt, weight))
            elif rule_type == "implies":
                self.implication_pairs.append((src, tgt, weight))
            elif rule_type == "maps_to_shared_concept":
                self.mapping_pairs.append((src, tgt, weight))
            elif rule_type == "risk_order":
                self.risk_order_pairs.append((src, tgt, weight))

    def compute(
        self,
        concept_logits: dict[str, torch.Tensor],
        concept_valid_masks: dict[str, torch.Tensor],
        node_id_mapping: dict[str, str | GraphNodeBinding] | None = None,
    ) -> torch.Tensor:
        """Compute graph consistency loss from concept logits.

        Args:
            concept_logits: Dict of head_name -> logits tensor [B, C] or [B]
            concept_valid_masks: Dict of head_name -> valid_mask tensor [B]
            node_id_mapping: Optional mapping from head_name to graph node_id.
                           If None, head_name is used directly.

        Returns:
            Scalar consistency loss tensor.
        """
        if not self.conflict_pairs and not self.implication_pairs and not self.risk_order_pairs:
            return torch.tensor(0.0)

        device = next(iter(concept_logits.values())).device
        loss_terms: list[torch.Tensor] = []

        probs = _compute_logits_to_probs_static(concept_logits)

        conflict_loss = self._compute_conflict_loss(probs, concept_valid_masks, node_id_mapping)
        if conflict_loss is not None:
            loss_terms.append(conflict_loss)

        implication_loss = self._compute_implication_loss(probs, concept_valid_masks, node_id_mapping)
        if implication_loss is not None:
            loss_terms.append(implication_loss)

        risk_order_loss = self._compute_risk_order_loss(probs, concept_valid_masks, node_id_mapping)
        if risk_order_loss is not None:
            loss_terms.append(risk_order_loss)

        if not loss_terms:
            return torch.tensor(0.0, device=device)

        return torch.stack(loss_terms).mean()

    def _resolve_node_probs(
        self,
        node_id: str,
        probs: dict[str, torch.Tensor],
        valid_masks: dict[str, torch.Tensor],
        node_id_mapping: dict[str, str | GraphNodeBinding] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Find the probability tensor and valid mask for a given graph node_id."""
        binding = _resolve_binding(node_id, node_id_mapping)
        head_name = str(binding.get("head_name", node_id))
        if head_name in probs and head_name in valid_masks:
            selected, _ = _select_node_probability(probs[head_name], binding)
            return selected, valid_masks[head_name]

        return None, None

    def _compute_conflict_loss(
        self,
        probs: dict[str, torch.Tensor],
        valid_masks: dict[str, torch.Tensor],
        node_id_mapping: dict[str, str | GraphNodeBinding] | None,
    ) -> torch.Tensor | None:
        terms: list[torch.Tensor] = []
        weights: list[float] = []

        for src_id, tgt_id, weight in self.conflict_pairs:
            src_prob, src_mask = self._resolve_node_probs(src_id, probs, valid_masks, node_id_mapping)
            tgt_prob, tgt_mask = self._resolve_node_probs(tgt_id, probs, valid_masks, node_id_mapping)

            if src_prob is None or tgt_prob is None:
                continue

            if src_mask is not None and tgt_mask is not None:
                both_valid = (src_mask > 0.5) & (tgt_mask > 0.5)
            elif src_mask is not None:
                both_valid = src_mask > 0.5
            elif tgt_mask is not None:
                both_valid = tgt_mask > 0.5
            else:
                both_valid = torch.ones_like(src_prob, dtype=torch.bool)

            if not both_valid.any():
                continue

            conflict_penalty = (src_prob * tgt_prob * both_valid.float()).sum() / both_valid.float().sum().clamp_min(1.0)
            terms.append(conflict_penalty)
            weights.append(weight)

        if not terms:
            return None

        weighted = torch.stack([t * w for t, w in zip(terms, weights)])
        return weighted.mean()

    def _compute_implication_loss(
        self,
        probs: dict[str, torch.Tensor],
        valid_masks: dict[str, torch.Tensor],
        node_id_mapping: dict[str, str | GraphNodeBinding] | None,
    ) -> torch.Tensor | None:
        terms: list[torch.Tensor] = []
        weights: list[float] = []

        for src_id, tgt_id, weight in self.implication_pairs:
            src_prob, src_mask = self._resolve_node_probs(src_id, probs, valid_masks, node_id_mapping)
            tgt_prob, tgt_mask = self._resolve_node_probs(tgt_id, probs, valid_masks, node_id_mapping)

            if src_prob is None or tgt_prob is None:
                continue

            if src_mask is not None and tgt_mask is not None:
                both_valid = (src_mask > 0.5) & (tgt_mask > 0.5)
            elif src_mask is not None:
                both_valid = src_mask > 0.5
            elif tgt_mask is not None:
                both_valid = tgt_mask > 0.5
            else:
                both_valid = torch.ones_like(src_prob, dtype=torch.bool)

            if not both_valid.any():
                continue

            violation = torch.relu(src_prob - tgt_prob) * both_valid.float()
            penalty = violation.sum() / both_valid.float().sum().clamp_min(1.0)
            terms.append(penalty)
            weights.append(weight)

        if not terms:
            return None

        weighted = torch.stack([t * w for t, w in zip(terms, weights)])
        return weighted.mean()

    def _compute_risk_order_loss(
        self,
        probs: dict[str, torch.Tensor],
        valid_masks: dict[str, torch.Tensor],
        node_id_mapping: dict[str, str | GraphNodeBinding] | None,
    ) -> torch.Tensor | None:
        terms: list[torch.Tensor] = []
        weights: list[float] = []

        for lower_id, higher_id, weight in self.risk_order_pairs:
            lower_prob, lower_mask = self._resolve_node_probs(lower_id, probs, valid_masks, node_id_mapping)
            higher_prob, higher_mask = self._resolve_node_probs(higher_id, probs, valid_masks, node_id_mapping)

            if lower_prob is None or higher_prob is None:
                continue

            if lower_mask is not None and higher_mask is not None:
                both_valid = (lower_mask > 0.5) & (higher_mask > 0.5)
            elif lower_mask is not None:
                both_valid = lower_mask > 0.5
            elif higher_mask is not None:
                both_valid = higher_mask > 0.5
            else:
                both_valid = torch.ones_like(lower_prob, dtype=torch.bool)

            if not both_valid.any():
                continue

            violation = torch.relu(lower_prob - higher_prob) * both_valid.float()
            penalty = violation.sum() / both_valid.float().sum().clamp_min(1.0)
            terms.append(penalty)
            weights.append(weight)

        if not terms:
            return None

        weighted = torch.stack([t * w for t, w in zip(terms, weights)])
        return weighted.mean()


def compute_graph_consistency_loss(
    concept_logits: dict[str, torch.Tensor],
    concept_valid_masks: dict[str, torch.Tensor],
    consistency_rules_path: str | Path,
    node_id_mapping: dict[str, str | GraphNodeBinding] | None = None,
    node_observed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convenience function for computing graph consistency loss.

    Only computes on observed_mask=1 entries. Missing is never negative.
    This is NOT a graph node direct alignment loss.
    """
    loss, _ = compute_graph_encoder_consistency_loss(
        concept_logits=concept_logits,
        concept_valid_masks=concept_valid_masks,
        consistency_rules_path=consistency_rules_path,
        graph_encoder_output=None,
        node_id_mapping=node_id_mapping,
        node_observed_mask=node_observed_mask,
    )
    return loss


def compute_graph_prototype_alignment_loss(
    concept_feature: torch.Tensor,
    graph_prototypes: dict[str, torch.Tensor],
    concept_targets: dict[str, torch.Tensor],
    concept_valid_masks: dict[str, torch.Tensor],
    prototype_projection: torch.nn.ModuleDict | None = None,
    prototype_alignment_metrics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> torch.Tensor:
    """Encourage concept features to align with graph-derived concept prototypes.

    For each active concept head, graph prototypes encode the ideal embedding
    for each class value. This loss penalizes the distance between a sample's
    concept feature and the prototype corresponding to its ground-truth label.

    Args:
        concept_feature: [B, D] shared concept feature (feeds all concept heads).
        graph_prototypes: dict head_name -> [C, H] prototype embeddings per class.
        concept_targets: dict head_name -> [B] or [B, C] target labels.
        concept_valid_masks: dict head_name -> [B] valid mask.
        prototype_projection: optional per-head Linear(D, H) to project concept
            feature into prototype space.
        prototype_alignment_metrics: optional dict updated with skipped/aligned
            prototype head counts.
        warnings: optional list updated when a head is skipped for unsafe
            dimensional alignment.

    Returns:
        Scalar prototype alignment loss, or zero if no valid samples.
    """
    loss_terms: list[torch.Tensor] = []
    weight_sum = 0.0
    metrics = prototype_alignment_metrics
    if metrics is not None:
        metrics["graph_prototype_head_count"] = int(len(graph_prototypes))
        metrics.setdefault("graph_prototype_aligned_head_count", 0)
        metrics.setdefault("graph_prototype_skipped_missing_label_count", 0)
        metrics.setdefault("graph_prototype_skipped_missing_projector_count", 0)
        metrics.setdefault("graph_prototype_skipped_projection_dim_count", 0)

    for head_name, prototypes in graph_prototypes.items():
        targets = concept_targets.get(head_name)
        masks = concept_valid_masks.get(head_name)
        if targets is None or masks is None:
            if metrics is not None:
                metrics["graph_prototype_skipped_missing_label_count"] += 1
            continue
        if not bool(masks.any()):
            continue

        valid_mask = masks.to(dtype=torch.bool, device=concept_feature.device)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            continue

        projected = concept_feature[valid_indices]
        prototypes_dev = prototypes.to(device=concept_feature.device, dtype=projected.dtype)
        projected_dim = int(projected.shape[-1])
        prototype_dim = int(prototypes_dev.shape[-1])
        if projected_dim != prototype_dim:
            proj = None
            if prototype_projection is not None and head_name in prototype_projection:
                proj = prototype_projection[head_name]
            if proj is None:
                if metrics is not None:
                    metrics["graph_prototype_skipped_missing_projector_count"] += 1
                if warnings is not None:
                    warnings.append(
                        "graph_prototype_alignment_skipped:"
                        f"head={head_name}:concept_dim={projected_dim}:"
                        f"prototype_dim={prototype_dim}:projector_missing"
                    )
                continue
            proj.to(device=projected.device, dtype=projected.dtype)
            projected = proj(projected)
            projected_dim = int(projected.shape[-1])
            if projected_dim != prototype_dim:
                if metrics is not None:
                    metrics["graph_prototype_skipped_projection_dim_count"] += 1
                if warnings is not None:
                    warnings.append(
                        "graph_prototype_alignment_skipped:"
                        f"head={head_name}:projected_dim={projected_dim}:"
                        f"prototype_dim={prototype_dim}:projector_dim_mismatch"
                    )
                continue
        if metrics is not None:
            metrics["graph_prototype_aligned_head_count"] += 1

        if head_name == "cancer_label":
            selected_targets = targets[valid_indices].to(device=concept_feature.device).flatten()
            idx_pos = torch.nonzero(selected_targets > 0.5, as_tuple=False).flatten()
            idx_neg = torch.nonzero(selected_targets <= 0.5, as_tuple=False).flatten()
            if idx_pos.numel() > 0 and prototypes_dev.shape[0] > 0:
                proto_pos = prototypes_dev[0]
                diffs_pos = (projected[idx_pos] - proto_pos).pow(2).mean(dim=-1)
                loss_terms.append(diffs_pos.mean())
                weight_sum += 1.0
            if idx_neg.numel() > 0:
                loss_terms.append(projected[idx_neg].new_zeros(()))
            continue

        if head_name == "finding":
            selected_targets = targets[valid_indices].to(device=concept_feature.device)
            if selected_targets.shape[1] != prototypes_dev.shape[0]:
                continue
            active_mask = selected_targets > 0.5
            for cls_idx in range(min(prototypes_dev.shape[0], int(selected_targets.shape[1]))):
                cls_valid = active_mask[:, cls_idx]
                if not cls_valid.any():
                    continue
                proto = prototypes_dev[cls_idx]
                diffs = (projected[cls_valid] - proto).pow(2).mean(dim=-1)
                loss_terms.append(diffs.mean())
                weight_sum += 1.0 / max(1.0, float(active_mask.sum(dim=1).clamp_min(1.0).mean().item()))
            continue

        selected_targets = targets[valid_indices].to(device=concept_feature.device, dtype=torch.long)
        for cls_idx in range(int(prototypes_dev.shape[0])):
            cls_valid = selected_targets == cls_idx
            if not cls_valid.any():
                continue
            proto = prototypes_dev[cls_idx]
            diffs = (projected[cls_valid] - proto).pow(2).mean(dim=-1)
            loss_terms.append(diffs.mean())
            weight_sum += 1.0

    if not loss_terms or weight_sum <= 0.0:
        return concept_feature.new_zeros(())

    return torch.stack(loss_terms).sum() / max(weight_sum, 1e-6)


def compute_graph_encoder_consistency_loss(
    concept_logits: dict[str, torch.Tensor],
    concept_valid_masks: dict[str, torch.Tensor],
    consistency_rules_path: str | Path,
    graph_encoder_output: Any | None = None,
    node_id_mapping: dict[str, str | GraphNodeBinding] | None = None,
    node_observed_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Graph consistency loss with per-rule node-pair prior weighting.

    When graph_encoder_output is provided, each consistency rule is weighted
    by the pairwise semantic prior of its source and target graph nodes.
    This replaces the weak global prior.mean() multiplier with rule-level
    node-pair prior lookup.

    Args:
        concept_logits: Head logits.
        concept_valid_masks: Head valid masks.
        consistency_rules_path: Path to consistency rules YAML.
        graph_encoder_output: Optional GraphEncoderOutput from graph encoder.
        node_id_mapping: Optional node_id -> head_name mapping.

    Returns:
        Tuple of (scalar consistency loss, metrics dict).
    """
    metrics: dict[str, Any] = {
        "graph_rule_term_count": 0,
        "graph_rule_skipped_count": 0,
        "graph_rule_skipped_reason_counts": {},
        "graph_pair_prior_mean": 0.0,
        "graph_encoder_consumed": graph_encoder_output is not None,
    }
    device = next(iter(concept_logits.values())).device

    # Build node_id -> graph_index lookup from graph encoder output
    node_id_to_graph_idx: dict[str, int] = {}
    if graph_encoder_output is not None:
        node_ids = getattr(graph_encoder_output, "node_ids", ())
        if node_ids:
            node_id_to_graph_idx = {nid: i for i, nid in enumerate(node_ids)}
        prior = getattr(graph_encoder_output, "graph_semantic_prior_matrix", None)
    else:
        prior = None

    observed_graph_nodes = node_observed_mask
    if observed_graph_nodes is None and graph_encoder_output is not None:
        observed_graph_nodes = getattr(graph_encoder_output, "node_observed_mask", None)
    if observed_graph_nodes is not None:
        if observed_graph_nodes.ndim != 2:
            raise ValueError("Graph observed node mask must have shape [B, N].")
        observed_graph_nodes = observed_graph_nodes.to(device=device, dtype=torch.bool)

    # Load rules for per-rule node-pair prior lookup
    rules_path = Path(consistency_rules_path)
    try:
        rules_raw = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read clinical graph consistency rules: {rules_path}"
        ) from exc

    raw_rules = _normalized_consistency_rules(rules_raw)
    if not raw_rules:
        return torch.tensor(0.0, device=next(iter(concept_logits.values())).device), metrics

    probs = _compute_logits_to_probs_static(concept_logits)
    loss_terms: list[torch.Tensor] = []
    pair_priors: list[float] = []
    term_count = 0
    skipped_count = 0
    skipped_reason_counts: dict[str, int] = {}

    def _skip(reason: str) -> None:
        nonlocal skipped_count
        skipped_count += 1
        skipped_reason_counts[reason] = skipped_reason_counts.get(reason, 0) + 1

    for rule in raw_rules:
        rule_type = rule.get("rule_type", "")
        src_id = rule.get("source_node_id", "")
        tgt_id = rule.get("target_node_id", "")
        base_weight = float(rule.get("penalty_weight", 1.0))

        if not src_id or not tgt_id:
            _skip("missing_rule_endpoint")
            continue

        src_binding = _resolve_binding(src_id, node_id_mapping)
        tgt_binding = _resolve_binding(tgt_id, node_id_mapping)

        src_head = str(src_binding.get("head_name", ""))
        tgt_head = str(tgt_binding.get("head_name", ""))
        if src_head not in probs and tgt_head not in probs:
            _skip("source_and_target_head_unmapped_or_inactive")
            continue
        if src_head not in probs:
            _skip("source_head_unmapped_or_inactive")
            continue
        if tgt_head not in probs:
            _skip("target_head_unmapped_or_inactive")
            continue

        src_prob, src_prob_skip = _select_node_probability(probs[src_head], src_binding)
        tgt_prob, tgt_prob_skip = _select_node_probability(probs[tgt_head], tgt_binding)
        if src_prob is None and tgt_prob is None:
            _skip(f"source_and_target_{src_prob_skip or tgt_prob_skip or 'probability_unavailable'}")
            continue
        if src_prob is None:
            _skip(f"source_{src_prob_skip or 'probability_unavailable'}")
            continue
        if tgt_prob is None:
            _skip(f"target_{tgt_prob_skip or 'probability_unavailable'}")
            continue
        src_mask = concept_valid_masks.get(src_head)
        tgt_mask = concept_valid_masks.get(tgt_head)

        if src_head == tgt_head:
            if src_mask is not None:
                both_valid = src_mask > 0.5
            elif tgt_mask is not None:
                both_valid = tgt_mask > 0.5
            else:
                both_valid = torch.ones_like(src_prob, dtype=torch.bool, device=device)
        elif src_mask is not None and tgt_mask is not None:
            both_valid = (src_mask > 0.5) & (tgt_mask > 0.5)
        elif src_mask is not None:
            both_valid = src_mask > 0.5
        elif tgt_mask is not None:
            both_valid = tgt_mask > 0.5
        else:
            both_valid = torch.ones_like(src_prob, dtype=torch.bool, device=device)

        src_graph_idx = node_id_to_graph_idx.get(src_id)
        tgt_graph_idx = node_id_to_graph_idx.get(tgt_id)
        if observed_graph_nodes is not None:
            if src_graph_idx is None or tgt_graph_idx is None:
                _skip("rule_endpoint_not_in_observed_graph_schema")
                continue
            if int(observed_graph_nodes.shape[0]) != int(both_valid.shape[0]):
                raise ValueError("Graph observed node mask batch size does not match concept logits.")
            both_valid = (
                both_valid
                & observed_graph_nodes[:, src_graph_idx]
                & observed_graph_nodes[:, tgt_graph_idx]
            )

        if not both_valid.any():
            _skip("no_confirmed_observed_mask_overlap")
            continue

        # Look up node-pair prior from graph semantic prior matrix
        if src_graph_idx is not None and tgt_graph_idx is not None and prior is not None:
            if prior.ndim == 3:
                pair_prior = prior[:, src_graph_idx, tgt_graph_idx]
            elif prior.ndim == 2:
                pair_prior = float(prior[src_graph_idx, tgt_graph_idx].detach().item())
            else:
                raise ValueError("Graph semantic prior matrix must have shape [N, N] or [B, N, N].")
        else:
            pair_prior = 0.5  # neutral prior if node not in graph

        if isinstance(pair_prior, torch.Tensor):
            pair_priors.append(float(pair_prior[both_valid].detach().mean().item()))
        else:
            pair_priors.append(pair_prior)

        # Per-rule node-pair prior weighting.
        #
        # pair_prior ∈ [0,1] is the (cosine_sim + 1)/2 entry from
        # graph_semantic_prior_matrix — it measures graph-structural relatedness
        # between the two nodes. Higher values mean the nodes are more
        # semantically/structurally related in the learned graph embedding space.
        #
        # dynamic_weight = base_weight * (0.5 + 0.5 * pair_prior)
        # maps pair_prior ∈ [0,1] → weight ∈ [0.5*base, 1.0*base].
        #
        # Per-rule-type intent:
        #   conflicts_with:  Higher weight for related-but-conflicting nodes
        #                    (hard negatives — the model must learn to separate
        #                    concepts that are similar in the graph but must
        #                    not co-occur clinically).
        #   implies:         Higher weight when antecedent and consequent are
        #                    graph-related (the implication is more semantically
        #                    grounded and must be enforced).
        #   risk_order:      Higher weight for related risk concepts (the
        #                    ordering constraint is more clinically meaningful).
        #   maps_to_shared:  Higher weight for related concepts that should
        #                    map to similar activations.
        #
        # This is NOT a global prior.mean() multiplier. Each rule gets its own
        # node-pair prior lookup from the graph encoder output. When the graph
        # encoder is unavailable, the neutral 0.5 prior keeps the same
        # class-specific rule path without falling back to legacy max(dim=1).
        if rule_type == "conflicts_with":
            raw_penalty = src_prob * tgt_prob
        elif rule_type == "implies":
            raw_penalty = torch.relu(src_prob - tgt_prob)
        elif rule_type == "risk_order":
            raw_penalty = torch.relu(src_prob - tgt_prob)
        elif rule_type == "maps_to_shared_concept":
            raw_penalty = (src_prob - tgt_prob) ** 2
        else:
            _skip("unsupported_rule_type")
            continue

        valid_float = both_valid.to(dtype=raw_penalty.dtype)
        if isinstance(pair_prior, torch.Tensor):
            dynamic_weight = base_weight * (0.5 + 0.5 * pair_prior.to(dtype=raw_penalty.dtype))
            penalty = (raw_penalty * valid_float * dynamic_weight).sum() / (
                valid_float * dynamic_weight
            ).sum().clamp_min(1.0e-6)
            loss_terms.append(penalty)
        else:
            dynamic_weight = base_weight * (0.5 + 0.5 * pair_prior)
            penalty = (raw_penalty * valid_float).sum() / valid_float.sum().clamp_min(1.0)
            loss_terms.append(penalty * dynamic_weight)
        term_count += 1

    metrics["graph_rule_term_count"] = term_count
    metrics["graph_rule_skipped_count"] = skipped_count
    metrics["graph_rule_skipped_reason_counts"] = skipped_reason_counts
    if pair_priors:
        metrics["graph_pair_prior_mean"] = float(sum(pair_priors) / len(pair_priors))

    if not loss_terms:
        return torch.tensor(0.0, device=device), metrics

    return torch.stack(loss_terms).mean(), metrics


def _compute_logits_to_probs_static(concept_logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    probs: dict[str, torch.Tensor] = {}
    for head_name, logits in concept_logits.items():
        if head_name == "finding":
            probs[head_name] = torch.sigmoid(logits)
        elif logits.dim() == 2 and logits.shape[1] > 1:
            probs[head_name] = torch.softmax(logits, dim=1)
        else:
            probs[head_name] = torch.sigmoid(logits.squeeze(-1))
    return probs


def _resolve_binding(
    node_id: str,
    node_id_mapping: dict[str, str | GraphNodeBinding] | None,
) -> GraphNodeBinding:
    """Resolve a graph node_id to a class-specific concept binding."""
    if node_id_mapping and node_id in node_id_mapping:
        raw = node_id_mapping[node_id]
        if isinstance(raw, dict):
            return {
                "head_name": str(raw.get("head_name", node_id)),
                "class_index": int(raw.get("class_index", 0)),
                "head_type": str(raw.get("head_type", "multiclass")),
            }
        return {"head_name": str(raw), "class_index": 0, "head_type": "multiclass"}
    return {"head_name": node_id, "class_index": 0, "head_type": "multiclass"}


def _select_node_probability(
    head_probs: torch.Tensor,
    binding: GraphNodeBinding,
) -> tuple[torch.Tensor | None, str | None]:
    head_type = str(binding.get("head_type", "multiclass"))
    class_index = int(binding.get("class_index", 0))
    if head_type == "binary":
        prob = head_probs.squeeze(-1)
        if class_index <= 0:
            return 1.0 - prob, None
        return prob, None
    if head_probs.dim() < 2:
        if class_index == 0:
            return head_probs.squeeze(-1), None
        return None, "class_index_out_of_range"
    if class_index < 0 or class_index >= int(head_probs.shape[1]):
        return None, "class_index_out_of_range"
    return head_probs[:, class_index], None
