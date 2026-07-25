from typing import Callable
from abc import abstractmethod
import caskade as ck
import jax.numpy as jnp
import jax
import cosmographi as cg


class SpartanLikelihood(ck.Module):
    def __init__(
        self,
        cosmology: cg.Cosmology,
        like_fn: Callable,
        like_fn_args: jnp.array,
        z_max: float = 2.0,
    ):
        super().__init__()
        self.cosmology = cosmology
        self.like_fn = like_fn  # f(z, mu, args) -> log_likelihood
        self.like_fn_args = like_fn_args  # (Nsn, Nargs)
        self.z_max = z_max
        self.vlike_fn = jax.jit(
            jax.vmap(jax.vmap(self.like_fn, in_axes=(None, None, 0)), in_axes=(0, 0, None))
        )

    @abstractmethod
    def logP_S_C(self):
        """Return the log marginal probability of the selection given cosmology."""
        pass

    @ck.forward
    def log_likelihood(self):
        z_model = jnp.linspace(0.001, self.z_max, 100)
        mu_model = 5 * jnp.log10(jax.vmap(self.cosmology.luminosity_distance)(z_model)) - 5
        ll = self.vlike_fn(z_model, mu_model, self.like_fn_args)
        ll = jnp.sum(jax.scipy.special.logsumexp(ll, axis=0))

        logP_S_C = self.like_fn_args.shape[0] * self.logP_S_C()

        return ll - logP_S_C

    @ck.forward
    def neg_log_likelihood(self):
        return -self.log_likelihood()
