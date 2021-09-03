from .losses import (
    torch_nll_gaussian,
    kl_categorical_uniform_direct,
    kl_categorical,
    cyc_anneal,
    get_prior_from_adj,
    get_simple_prior
)

__all__ = [
    "torch_nll_gaussian",
    "kl_categorical_uniform_direct",
    "kl_categorical",
    "cyc_anneal",
    "get_prior_from_adj",
    "get_simple_prior"
]

