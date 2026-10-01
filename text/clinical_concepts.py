from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_STAGE1_CONCEPT_HEADS = (
    "view",
    "laterality",
    "density",
    "finding",
    "birads",
    "cancer_label",
    "benign_malignant_label",
    "mri_sequence",
    "mri_enhancement",
    "mri_kinetic_curve",
    "mri_treatment_response",
    "us_shape",
    "us_margin",
    "us_echogenicity",
    "us_orientation",
    "us_posterior_feature",
    "us_vascularity",
)
VIEW_LABELS = ("CC", "MLO")
LATERALITY_LABELS = ("left", "right")
DENSITY_LABELS = ("A", "B", "C", "D")
BIRADS_LABELS = ("0", "1", "2", "3", "4", "4A", "4B", "4C", "5", "6")
FINDING_LABELS = (
    "no_finding",
    "mass",
    "calcification",
    "asymmetry",
    "architectural_distortion",
    "distortion",
    "other",
)
BENIGN_MALIGNANT_LABELS = ("benign", "malignant")
MRI_SEQUENCE_LABELS = ("dce", "t1", "t2", "dwi", "adc")
MRI_ENHANCEMENT_LABELS = ("mass", "non_mass")
MRI_KINETIC_CURVE_LABELS = ("persistent", "plateau", "washout")
MRI_TREATMENT_RESPONSE_LABELS = (
    "complete_response",
    "partial_response",
    "stable_disease",
    "progressive_disease",
    "residual_disease",
)
US_SHAPE_LABELS = ("oval", "round", "irregular")
US_MARGIN_LABELS = ("circumscribed", "not_circumscribed")
US_ECHOGENICITY_LABELS = ("anechoic", "hypoechoic", "isoechoic", "hyperechoic")
US_ORIENTATION_LABELS = ("parallel", "not_parallel")
US_POSTERIOR_FEATURE_LABELS = ("enhancement", "shadowing", "none")
US_VASCULARITY_LABELS = ("present", "absent")

_VIEW_PREFIX_LOOKUP = {
    "LCC": ("CC", "left"),
    "RCC": ("CC", "right"),
    "LMLO": ("MLO", "left"),
    "RMLO": ("MLO", "right"),
}
_VIEW_NORMALIZATION = {
    "CC": "CC",
    "MLO": "MLO",
}
_LATERALITY_NORMALIZATION = {
    "L": "left",
    "LEFT": "left",
    "LT": "left",
    "R": "right",
    "RIGHT": "right",
    "RT": "right",
}
_DENSITY_NORMALIZATION = {
    "A": "A",
    "DENSITY A": "A",
    "ALMOST ENTIRELY FATTY": "A",
    "B": "B",
    "DENSITY B": "B",
    "SCATTERED FIBROGLANDULAR DENSITIES": "B",
    "C": "C",
    "DENSITY C": "C",
    "HETEROGENEOUSLY DENSE": "C",
    "D": "D",
    "DENSITY D": "D",
    "EXTREMELY DENSE": "D",
}
_FINDING_NORMALIZATION = {
    "NO FINDING": "no_finding",
    "NO_FINDING": "no_finding",
    "NEGATIVE": "no_finding",
    "MASS": "mass",
    "CALCIFICATION": "calcification",
    "CALCIFICATIONS": "calcification",
    "SUSPICIOUS CALCIFICATION": "calcification",
    "ASYMMETRY": "asymmetry",
    "FOCAL ASYMMETRY": "asymmetry",
    "GLOBAL ASYMMETRY": "asymmetry",
    "ARCHITECTURAL DISTORTION": "architectural_distortion",
    "DISTORTION": "distortion",
}
_BENIGN_MALIGNANT_NORMALIZATION = {
    "BENIGN": "benign",
    "MALIGNANT": "malignant",
}
_MRI_SEQUENCE_NORMALIZATION = {
    "DCE": "dce",
    "DYNAMIC CONTRAST ENHANCED": "dce",
    "CONTRAST ENHANCED": "dce",
    "DYN": "dce",
    "DYNAMIC": "dce",
    "VIBRANT": "dce",
    "MULTIPHASE": "dce",
    "POST CONTRAST": "dce",
    "T1": "t1",
    "T1W": "t1",
    "T1 WEIGHTED": "t1",
    "T2": "t2",
    "T2W": "t2",
    "T2 WEIGHTED": "t2",
    "DWI": "dwi",
    "DIFFUSION": "dwi",
    "DIFFUSION WEIGHTED": "dwi",
    "ADC": "adc",
    "ADC MAP": "adc",
}

# Tokens that signal DCE in compound series descriptions (e.g. "ax dyn pre").
_DCE_SIGNAL_TOKENS = frozenset({"DCE", "DYN", "DYNAMIC", "VIBRANT", "MULTIPHASE", "CONTRAST"})

# Tokens that block DCE inference (scout / localizer / calibration scans).
_DCE_EXCLUDE_TOKENS = frozenset({"SCOUT", "LOCALIZER", "LOCATOR", "CALIBRATION", "SURVEY", "PILOT"})
_MRI_TREATMENT_RESPONSE_NORMALIZATION = {
    "COMPLETE RESPONSE": "complete_response",
    "CR": "complete_response",
    "PCR": "complete_response",
    "PATHOLOGICAL COMPLETE RESPONSE": "complete_response",
    "PARTIAL RESPONSE": "partial_response",
    "PR": "partial_response",
    "STABLE DISEASE": "stable_disease",
    "SD": "stable_disease",
    "STABLE": "stable_disease",
    "PROGRESSIVE DISEASE": "progressive_disease",
    "PD": "progressive_disease",
    "PROGRESSION": "progressive_disease",
    "RESIDUAL DISEASE": "residual_disease",
    "RD": "residual_disease",
    "RESIDUAL": "residual_disease",
}
_BINARY_TRUE_VALUES = {"1", "TRUE", "YES", "Y", "POSITIVE", "CANCER", "MALIGNANT"}
_BINARY_FALSE_VALUES = {"0", "FALSE", "NO", "N", "NEGATIVE", "BENIGN"}


@dataclass(frozen=True)
class ClinicalConceptSample:
    modality: str
    view: str
    laterality: str
    density: str
    finding: str
    birads: str
    cancer_label: str
    benign_malignant_label: str
    study_description: str


def _normalize_string(raw_value: Any) -> str:
    return str(raw_value or "").strip()


def _normalize_prompt_value(raw_value: str) -> str:
    return raw_value if raw_value else "unknown"


def _canonical_upper(raw_value: Any) -> str:
    return re.sub(r"\s+", " ", _normalize_string(raw_value).upper().replace("-", " ").replace("_", " ")).strip()


def normalize_laterality(raw_value: Any) -> str:
    normalized = _canonical_upper(raw_value)
    return _LATERALITY_NORMALIZATION.get(normalized, "")


def normalize_view(raw_value: Any) -> tuple[str, str | None]:
    normalized = _normalize_string(raw_value).upper().replace("-", "").replace("_", "").replace(" ", "")
    if not normalized:
        return "", None
    if normalized in _VIEW_PREFIX_LOOKUP:
        return _VIEW_PREFIX_LOOKUP[normalized]
    if normalized.startswith("L") and normalized[1:] in _VIEW_NORMALIZATION:
        return _VIEW_NORMALIZATION[normalized[1:]], "left"
    if normalized.startswith("R") and normalized[1:] in _VIEW_NORMALIZATION:
        return _VIEW_NORMALIZATION[normalized[1:]], "right"
    return _VIEW_NORMALIZATION.get(normalized, ""), None


def normalize_density(raw_value: Any) -> str:
    normalized = _canonical_upper(raw_value)
    if not normalized or normalized == "UNKNOWN":
        return ""
    return _DENSITY_NORMALIZATION.get(normalized, "")


def normalize_birads(raw_value: Any) -> str:
    text = _canonical_upper(raw_value)
    if not text or text == "UNKNOWN":
        return ""
    text = re.sub(r"BI\s*RADS", " ", text)
    text = text.replace("BIRADS", " ")
    text = re.sub(r"\s+", "", text)
    match = re.search(r"(4A|4B|4C|[0-6])", text)
    if match is None:
        return ""
    normalized = match.group(1).upper()
    return normalized if normalized in BIRADS_LABELS else ""


def _normalize_finding_token(raw_value: str) -> str:
    normalized = re.sub(r"\s+", " ", raw_value.strip().upper().replace("-", " ").replace("_", " "))
    if not normalized or normalized == "UNKNOWN":
        return ""
    if normalized in _FINDING_NORMALIZATION:
        return _FINDING_NORMALIZATION[normalized]
    if "DISTORTION" in normalized and "ARCHITECTURAL" in normalized:
        return "architectural_distortion"
    if "DISTORTION" in normalized:
        return "distortion"
    if "ASYMMETRY" in normalized:
        return "asymmetry"
    if "CALCIFICATION" in normalized:
        return "calcification"
    if "MASS" in normalized:
        return "mass"
    return "other"


def normalize_finding_labels(raw_value: Any) -> tuple[str, ...]:
    raw_text = _normalize_string(raw_value)
    if not raw_text or raw_text.lower() == "unknown":
        return ()
    try:
        literal = ast.literal_eval(raw_text)
        if isinstance(literal, list):
            raw_items = [str(item) for item in literal]
        else:
            raw_items = [raw_text]
    except (SyntaxError, ValueError):
        raw_items = [part.strip() for part in re.split(r"[|;/,]+", raw_text) if part.strip()]
    labels: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        normalized = _normalize_finding_token(raw_item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        labels.append(normalized)
    return tuple(labels)


def normalize_cancer_label(raw_value: Any) -> str:
    normalized = _canonical_upper(raw_value)
    if not normalized or normalized == "UNKNOWN":
        return ""
    if normalized in _BINARY_TRUE_VALUES:
        return "1"
    if normalized in _BINARY_FALSE_VALUES:
        return "0"
    return ""


def normalize_benign_malignant_label(raw_value: Any) -> str:
    normalized = _canonical_upper(raw_value)
    if not normalized or normalized == "UNKNOWN":
        return ""
    return _BENIGN_MALIGNANT_NORMALIZATION.get(normalized, "")


def _match_mri_sequence_by_keyword(normalized: str) -> str:
    """Keyword-based fallback for compound series descriptions (e.g. 'ax dyn pre').

    Priority (highest first):
      1. DWI / ADC — explicit non-DCE sequences
      2. T2 — before T1 (more specific)
      3. T1 — before DCE
      4. DCE signals (dyn / dynamic / vibrant / multiphase / contrast / phase numbering)
         unless blocked by scout / localizer / calibration / survey / pilot tokens.
    """
    tokens = set(normalized.split())

    # ── dwi / adc: highest non-dce priority ──
    if "DWI" in tokens or "DIFFUSION" in tokens:
        return "dwi"
    if "ADC" in tokens:
        return "adc"

    # ── t2: before t1 (more specific) ──
    if "T2" in tokens or "T2W" in tokens:
        return "t2"

    # ── t1: before dce ──
    if "T1" in tokens or "T1W" in tokens:
        return "t1"

    # ── exclusion: scout / localizer / calibration / survey / pilot block dce ──
    if tokens & _DCE_EXCLUDE_TOKENS:
        return ""

    # ── dce signal tokens ──
    if tokens & _DCE_SIGNAL_TOKENS:
        return "dce"

    # ── multi + phase → multiphase dynamic ──
    if "MULTI" in tokens and "PHASE" in tokens:
        return "dce"

    # ── phase numbering: PH1 / PH2 / PH3 / PH4 (and compounds like PH2AX) ──
    for token in tokens:
        if re.match(r"^PH[1-4]", token):
            return "dce"

    # ── subtraction with contrast / dynamic context → dce ──
    if "SUBTRACTION" in tokens and (
        "CONTRAST" in tokens or "POST" in tokens or "DYN" in tokens or "DYNAMIC" in tokens
    ):
        return "dce"

    return ""


def normalize_mri_sequence(raw_value: Any) -> str:
    normalized = _canonical_upper(raw_value)
    if not normalized or normalized == "UNKNOWN":
        return ""
    result = _MRI_SEQUENCE_NORMALIZATION.get(normalized, "")
    if result:
        return result
    return _match_mri_sequence_by_keyword(normalized)


def normalize_mri_treatment_response(raw_value: Any) -> str:
    normalized = _canonical_upper(raw_value)
    if not normalized or normalized == "UNKNOWN":
        return ""
    return _MRI_TREATMENT_RESPONSE_NORMALIZATION.get(normalized, "")


def build_clinical_concepts(sample: dict[str, Any]) -> ClinicalConceptSample:
    modality = str(sample.get("modality", "")).strip().lower()
    normalized_modality = "mammography" if modality == "mammo" else (modality or "unknown")

    view, inferred_laterality = normalize_view(sample.get("view"))
    laterality = normalize_laterality(sample.get("laterality"))
    if not laterality and inferred_laterality is not None:
        laterality = inferred_laterality

    density = normalize_density(sample.get("density"))
    finding_labels = normalize_finding_labels(sample.get("finding"))
    birads = normalize_birads(sample.get("birads"))
    cancer_label = normalize_cancer_label(sample.get("cancer_label"))
    benign_malignant_label = normalize_benign_malignant_label(
        sample.get("benign_malignant_label")
    )

    return ClinicalConceptSample(
        modality=normalized_modality,
        view=_normalize_prompt_value(view),
        laterality=_normalize_prompt_value(laterality),
        density=_normalize_prompt_value(density),
        finding=_normalize_prompt_value("|".join(finding_labels)),
        birads=_normalize_prompt_value(birads),
        cancer_label=_normalize_prompt_value(cancer_label),
        benign_malignant_label=_normalize_prompt_value(benign_malignant_label),
        study_description=_normalize_prompt_value(_normalize_string(sample.get("study_description"))),
    )


def build_case_specific_structured_prompt(concepts: ClinicalConceptSample) -> str:
    return (
        "Structured mammography concept prompt: "
        f"modality={concepts.modality}; "
        f"view={concepts.view}; "
        f"laterality={concepts.laterality}; "
        f"density={concepts.density}; "
        f"finding={concepts.finding}."
    )


def build_stage1_structured_prompt(concepts: ClinicalConceptSample) -> str:
    return build_case_specific_structured_prompt(concepts)


def build_stage2_structured_prompt(concepts: ClinicalConceptSample) -> str:
    return build_case_specific_structured_prompt(concepts)


def view_to_index(view: str) -> int:
    return VIEW_LABELS.index(view)


def laterality_to_index(laterality: str) -> int:
    return LATERALITY_LABELS.index(laterality)


def density_to_index(density: str) -> int:
    return DENSITY_LABELS.index(density)


def birads_to_index(birads: str) -> int:
    return BIRADS_LABELS.index(birads)


def benign_malignant_to_index(label: str) -> int:
    return BENIGN_MALIGNANT_LABELS.index(label)


def mri_sequence_to_index(label: str) -> int:
    return MRI_SEQUENCE_LABELS.index(label)


def mri_treatment_response_to_index(label: str) -> int:
    return MRI_TREATMENT_RESPONSE_LABELS.index(label)


def finding_labels_to_multi_hot(labels: tuple[str, ...]) -> list[float]:
    vector = [0.0] * len(FINDING_LABELS)
    for label in labels:
        if label not in FINDING_LABELS:
            continue
        vector[FINDING_LABELS.index(label)] = 1.0
    return vector


def concept_head_output_dim(head_name: str) -> int:
    if head_name == "view":
        return len(VIEW_LABELS)
    if head_name == "laterality":
        return len(LATERALITY_LABELS)
    if head_name == "density":
        return len(DENSITY_LABELS)
    if head_name == "birads":
        return len(BIRADS_LABELS)
    if head_name == "finding":
        return len(FINDING_LABELS)
    if head_name == "cancer_label":
        return 1
    if head_name == "benign_malignant_label":
        return len(BENIGN_MALIGNANT_LABELS)
    if head_name == "mri_sequence":
        return len(MRI_SEQUENCE_LABELS)
    if head_name == "mri_enhancement":
        return len(MRI_ENHANCEMENT_LABELS)
    if head_name == "mri_kinetic_curve":
        return len(MRI_KINETIC_CURVE_LABELS)
    if head_name == "mri_treatment_response":
        return len(MRI_TREATMENT_RESPONSE_LABELS)
    if head_name == "us_shape":
        return len(US_SHAPE_LABELS)
    if head_name == "us_margin":
        return len(US_MARGIN_LABELS)
    if head_name == "us_echogenicity":
        return len(US_ECHOGENICITY_LABELS)
    if head_name == "us_orientation":
        return len(US_ORIENTATION_LABELS)
    if head_name == "us_posterior_feature":
        return len(US_POSTERIOR_FEATURE_LABELS)
    if head_name == "us_vascularity":
        return len(US_VASCULARITY_LABELS)
    raise KeyError(f"Unsupported concept head: {head_name}")
