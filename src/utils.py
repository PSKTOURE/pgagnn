import os
import random
from functools import cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_data_dir = Path(__file__).resolve().parent / "data"

INNER_PRODUCT_INDICES = torch.tensor([1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=torch.bool)
INFINITE_NORM_INDICES = ~INNER_PRODUCT_INDICES

_FILENAMES = {
    "gp": str((_data_dir / "geometric_product.pt").resolve()),
    "op": str((_data_dir / "outer_product.pt").resolve()),
}
_device = torch.device("cpu")

basis_elements = [
    [(0, 0)],  # Grade 0: scalar → scalar
    [(1, 1), (2, 2), (3, 3), (4, 4)],  # Grade 1: vectors → vectors
    [
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
        (10, 10),
    ],  # Grade 2: bivectors → bivectors
    [(11, 11), (12, 12), (13, 13), (14, 14)],  # Grade 3: trivectors → trivectors
    [(15, 15)],  # Grade 4: pseudoscalar → pseudoscalar
    [(1, 0)],  # e0 × scalar → vector component 1
    [(5, 2), (6, 3), (7, 4)],  # e0 × vector → bivector
    [(11, 8), (12, 9), (13, 10)],  # e0 × bivector → trivector
    [(15, 14)],  # e0 × trivector → pseudoscalar
]


@cache
def _compute_pin_equilinear_basis(device=_device, dtype=torch.float32, normalize: bool = True):
    """Constructs basis elements for Pin(3,0,1)-equivariant linear maps between multivectors.

    This function is cached.

    Returns
    -------
    basis : torch.Tensor with shape (9, 16, 16)
        Basis elements for equivariant linear maps.
    """
    basis = torch.zeros((len(basis_elements), 16, 16), device=device, dtype=dtype)
    for idx, elements in enumerate(basis_elements):
        w = torch.zeros((16, 16), device=device, dtype=dtype)
        for i, j in elements:
            w[i, j] = 1.0
        if normalize:
            w = F.normalize(w, p=2, dim=(0, 1))
        basis[idx] = w
    return basis


@cache
def _load_bilinear_basis(kind: str, device=_device, dtype=torch.float32) -> torch.Tensor:
    """Loads basis elements for Pin-equivariant bilinear maps between multivectors.

    Parameters
    ----------
    kind : {"gp", "op"}
        Filename of the basis file
    device : torch.Device or str
        Device
    dtype : torch.Dtype
        Data type

    Returns
    -------
    basis : torch.Tensor with shape (num_basis_elements, 16, 16, 16)
        Basis elements for bilinear equivariant maps between multivectors.
    """

    filename = _FILENAMES[kind]
    sparse_basis = torch.load(filename).to(torch.float32)
    # Convert to dense tensor
    # The reason we do that is that einsum is not defined for sparse tensors
    basis = sparse_basis.to_dense()

    return basis.to(device=device, dtype=dtype)



@cache
def _compute_reversal(device=_device, dtype=torch.float32) -> torch.Tensor:
    """Constructs a matrix that computes multivector reversal.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    reversal_diag : torch.Tensor with shape (16,)
        The diagonal of the reversal matrix, consisting of +1 and -1 entries.
    """
    reversal_flat = torch.ones(16, device=device, dtype=dtype)
    reversal_flat[5:15] = -1
    return reversal_flat


@torch.no_grad()
@cache
def _compute_dualization(device=_device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs a tensor for the dual operation.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    permutation : list of int
        Permutation index list to compute the dual
    factors : torch.Tensor
        Signs to multiply the dual outputs with.
    """
    permutation = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    factors = torch.tensor(
        [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1],
        device=device,
        dtype=dtype,
    )
    return permutation, factors


@torch.no_grad()
@cache
def _get_anti_dual_factors(device=_device, dtype=torch.float32) -> torch.Tensor:
    """Get factors for anti-dual operation.

    Parameters
    ----------
    device: torch.device
        Device.
    dtype: torch.dtype
        Dtype.

    Returns
    -------
    factors : Tensor with shape (16,)
        Factors for anti-dual operation.
    """
    factors = torch.ones(16, device=device, dtype=dtype)
    factors[[1, 2, 3, 4, 11, 12, 13, 14]] = -1
    return factors


def _build_dist_basis(device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute basis features for queries and keys in the geometric SDP attention.

    Parameters
    ----------
    device: torch.device
        Device.
    dtype: torch.dtype
        Dtype.

    Returns
    -------
    basis_q : Tensor with shape (4, 4, 5)
        Basis features for queries.
    basis_k : Tensor with shape (4, 4, 5)
        Basis features for keys.
    """
    r3 = torch.arange(3, device=device)
    basis_q = torch.zeros((4, 4, 5), device=device, dtype=dtype)
    basis_k = torch.zeros((4, 4, 5), device=device, dtype=dtype)

    # -sum_i (q_i^2) * k_0^2
    basis_q[r3, r3, 0] = 1
    basis_k[3, 3, 0] = -1

    # -q_0^2 * sum_i (k_i^2)
    basis_q[3, 3, 1] = 1
    basis_k[r3, r3, 1] = -1

    # sum_i 2 q_0 q_i k_0 k_i
    basis_q[r3, 3, 2 + r3] = 1
    basis_k[r3, 3, 2 + r3] = 2

    return basis_q, basis_k


def _build_dist_vec(tri: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Build 5D vector whose inner product with another such vector computes the squared distance.

    Parameters
    ----------
    tri: Tensor
        Batch of multivectors, only trivector part is used.
    basis: Tensor
        One of the bases from _build_dist_basis.

    Returns
    -------
    outputs : Tensor
        Batch of 5D vectors
    """
    e123 = tri[..., [3]]
    normalizer_factor = e123 / (torch.square(e123) + 1e-3)
    tri_normed = tri * normalizer_factor
    vec = torch.einsum("xyz,...x,...y->...z", basis, tri_normed, tri_normed)
    return vec


def parse_basis_element(element: str, basis_map: dict) -> tuple[float, int]:
    """Parses a single basis element term into its coefficient and index.

    Parameters
    ----------
    element : str
        Single basis element term, e.g., "2*e01", "e1", or "3"
    basis_map : dict
        Mapping from basis strings to indices

    Returns
    -------
    coeff : float
        Coefficient of the basis element.
    index : int
        Index of the basis element in the basis map.
    """
    element = element.strip()
    if "*" in element:
        coeff, basis = element.split("*")
        coeff = float(coeff)
    else:
        if element.startswith("e"):
            coeff = 1.0
            basis = element
        else:
            coeff = float(element)
            basis = ""

    index = basis_map[basis]
    return coeff, index


def parse_multivector(mv_str: str, basis_map: dict) -> torch.Tensor:
    """Parses a multivector string into a 16-dimensional tensor.

    Parameters
    ----------
    mv_str : str
        Multivector as string, e.g., "e1 + 2*e2 + 3*e3 + 1*e0"
    basis_map : dict
        Mapping from basis strings to indices

    Returns
    -------
    tensor : torch.Tensor
        16-dimensional tensor representing the multivector
    """
    tensor = torch.zeros(16)
    # Replace - with +- to handle negative terms, then split on +
    mv_str = mv_str.replace(" ", "").replace("-", "+-")
    terms = [t.strip() for t in mv_str.split("+") if t.strip()]

    for term in terms:
        coeff, index = parse_basis_element(term, basis_map)
        tensor[index] += coeff

    return tensor


def print_geometric_product(x: str, y: str, type_x: str, type_y: str) -> None:
    """Prints the geometric product of two multivectors.

    Parameters
    ----------
    x : str
        First multivector as string, e.g., "e1 + 2*e2 + 3*e3 + 1*e0"
    y : str
        Second multivector as string
    """
    basis_map = {
        "": 0,
        "e0": 1,
        "e1": 2,
        "e2": 3,
        "e3": 4,
        "e01": 5,
        "e02": 6,
        "e03": 7,
        "e12": 8,
        "e13": 9,
        "e23": 10,
        "e012": 11,
        "e013": 12,
        "e023": 13,
        "e123": 14,
        "e0123": 15,
    }
    reverse_basis_map = {v: k for k, v in basis_map.items()}

    coeff_x = parse_multivector(x, basis_map)
    coeff_y = parse_multivector(y, basis_map)
    gp_basis = _load_bilinear_basis("gp")

    # Compute geometric product: sum over i,j of gp_basis[k,i,j] * x[i] * y[j]
    gp = torch.einsum("kij,i,j->k", gp_basis, coeff_x, coeff_y)

    output_terms = []
    for i in range(16):
        if gp[i].abs() > 1e-6:
            basis_str = reverse_basis_map[i] if reverse_basis_map[i] else "1"
            output_terms.append(f"{float(gp[i].item()):.3f}*{basis_str}")

    output_str = " + ".join(output_terms) if output_terms else "0"
    output_str = output_str.replace("+ -", "- ")
    print(f"Geometric product of {type_x} ({x}) and {type_y} ({y}) is:\n  {output_str}\n")




def seed_everything(seed: int, deterministic: bool = True):
    """Seed all RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
