"""Projective Geometric Algebra Graph Neural Network (PGA-GNN) package."""

from .attention import (
    GADualNormAttention,
    GAGeometricAttention,
    GASimilarityAttention,
    GATrAttention,
    GATrAttentionSparse,
    get_attention_module,
)
from .checkpointing import (
    StopSignalHandler,
    load_checkpoint,
    save_checkpoint,
)
from .ggnn import (
    GGNN,
    GGNN_Block,
)
from .layers import (
    EquiLayerNorm,
    EquiLinear,
    GatedNonLinearity,
    GaussianRadialBasisLayer,
    GeometricBilinear,
    GradeDropout,
    ResidualLayer,
)
from .pgagnn import (
    PGA_GNN,
    EquiMLP,
    GA_MessageLayer,
    ScalarMLP,
)
from .primitives import (
    anti_dual,
    depthwise_equi_linear,
    dual,
    embed_oriented_plane,
    embed_pluecker_ray,
    embed_point,
    embed_scalar,
    embed_translation,
    embed_vector,
    equi_linear,
    equivariant_join,
    extract_point,
    geometric_product,
    grade_project,
    infinite_norm,
    inner_product,
    invariants,
    join,
    norm_squared,
    outer_product,
    reverse,
)
from .utils import seed_everything
from .xformers_stub import ensure_xformers_stub

# Automatically install xformers stub if xformers is unavailable or disabled
ensure_xformers_stub()

__all__ = [
    "GGNN",
    # Models & Blocks
    "PGA_GNN",
    "EquiLayerNorm",
    # Layers
    "EquiLinear",
    "EquiMLP",
    "GADualNormAttention",
    "GAGeometricAttention",
    "GASimilarityAttention",
    "GATrAttention",
    "GATrAttentionSparse",
    "GA_MessageLayer",
    "GGNN_Block",
    "GatedNonLinearity",
    "GaussianRadialBasisLayer",
    "GeometricBilinear",
    "GradeDropout",
    "ResidualLayer",
    "ScalarMLP",
    "StopSignalHandler",
    "anti_dual",
    "depthwise_equi_linear",
    "dual",
    "embed_oriented_plane",
    "embed_pluecker_ray",
    # Primitives & Embeddings
    "embed_point",
    "embed_scalar",
    "embed_translation",
    "embed_vector",
    "equi_linear",
    "equivariant_join",
    "extract_point",
    "geometric_product",
    # Attention Mechanisms
    "get_attention_module",
    "grade_project",
    "infinite_norm",
    "inner_product",
    "invariants",
    "join",
    "load_checkpoint",
    "norm_squared",
    "outer_product",
    "reverse",
    "save_checkpoint",
    # Utilities
    "seed_everything",
    "ensure_xformers_stub",
]
