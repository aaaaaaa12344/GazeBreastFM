from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breast_pretrain.train.stage1_joint import run_stage1_joint_trainer
from breast_pretrain.train.stage1_joint.prelaunch_control_plane import run_control_plane_once
from audit_final_model_compliance import build_report, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V6 formal Stage 1 joint trainer."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "configs" / "formal_stage1_v6_1.example.yaml",
        help="Path to the formal Stage 1 trainer YAML config.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Path to config (named alternative).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the output directory.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for the maximum number of training steps.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional override for the maximum number of dataset samples.",
    )
    parser.add_argument(
        "--backbone-weight-path",
        type=str,
        default=None,
        help="Path to stripped Mammo-FM checkpoint.",
    )
    parser.add_argument(
        "--text-dim",
        type=int,
        default=None,
        help="Text embedding dimension (resolved from prompt embeddings if not provided).",
    )
    parser.add_argument(
        "--init-state",
        type=Path,
        default=None,
        help="Path to formal_init_state.pt for immutable weight initialization.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Path to formal run artifact directory.",
    )
    parser.add_argument(
        "--high-conf-prior-npz",
        type=Path,
        default=None,
        help="Path to pre-materialized high_conf_patch_priors.npz.",
    )
    parser.add_argument(
        "--high-conf-prior-manifest",
        type=Path,
        default=None,
        help="Path to high_conf_patch_prior_manifest.csv.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config_path or args.config
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "formal_stage1_v6_1.example.yaml"
    if args.init_state is None:
        raise SystemExit("Formal Stage 1 requires --init-state before control-plane validation.")

    def _full_validation(path: Path) -> dict[str, object]:
        compliance = build_report(path)
        local_output = Path("/tmp/hsm_formal_stage1_prelaunch") / str(
            __import__("os").environ["HSM_FORMAL_PRELAUNCH_ID"]
        ) / "final_model_compliance"
        write_report(compliance, local_output)
        if compliance["is_final_config"] and compliance["status"] == "fail":
            raise RuntimeError("Formal Stage 1 final-model compliance failed.")
        return compliance

    run_control_plane_once(config_path, args.init_state, _full_validation)
    run_stage1_joint_trainer(
        config_path=config_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        backbone_weight_path=args.backbone_weight_path,
        text_dim=args.text_dim,
        init_state_path=args.init_state,
        artifact_dir=args.artifact_dir,
        high_conf_prior_npz=args.high_conf_prior_npz,
        high_conf_prior_manifest=args.high_conf_prior_manifest,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
