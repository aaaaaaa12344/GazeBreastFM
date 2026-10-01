from __future__ import annotations

from torch import nn

from breast_pretrain.text.clinical_concepts import concept_head_output_dim


class ClinicalConceptHeads(nn.Module):
    """Heads keyed by external concept id without leaking PyTorch key restrictions."""

    def __init__(
        self,
        input_dim: int,
        active_head_names: tuple[str, ...],
        output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self._module_names = {name: name.replace(".", "__") for name in active_head_names}
        self.heads = nn.ModuleDict(
            {
                self._module_names[head_name]: nn.Linear(
                    input_dim,
                    int(output_dims[head_name]) if output_dims and head_name in output_dims else concept_head_output_dim(head_name),
                )
                for head_name in active_head_names
            }
        )

    def forward(self, concept_feature):
        return {
            head_name: self.heads[module_name](concept_feature)
            for head_name, module_name in self._module_names.items()
        }
