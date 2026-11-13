from typing import Callable
from abc import abstractmethod
import caskade as ck
import jax.numpy as jnp
import jax
from jaxtyping import PyTree
import cosmographi as cg


class SpartanLikelihood(ck.Module):
    def __init__(
        self,
        cosmology: cg.Cosmology,
        ll_fn: Callable,
        ll_fn_args: PyTree,
        z_min: float = 0.001,
        z_max: float = 2.0,
        z_steps: int = 1000,
    ):
        super().__init__()
        self.cosmology = cosmology
        self.ll_fn = ll_fn  # f(z-mu vec, args) -> log_likelihood
        self.ll_fn_args = ll_fn_args  # PyTree, elements vectorized on first axis
        n_sn = list(jax.tree.map(lambda x: x.shape[0], ll_fn_args))
        assert all(
            n_sn[0] == n for n in n_sn
        ), "All elements in ll_fn_args must have the same length first dimension"
        self.n_sn = n_sn[0]
        self.z_min = z_min
        self.z_max = z_max
        self.z_steps = z_steps
        self.vll_fn = jax.jit(jax.vmap(jax.vmap(self.ll_fn, in_axes=(None, 0)), in_axes=(0, None)))

    @abstractmethod
    def logP_S_C(self):
        """Return the log marginal probability of the selection given cosmology."""
        pass

    @ck.forward
    def log_likelihood(self):
        z_model = jnp.linspace(self.z_min, self.z_max, self.z_steps)
        mu_model = 5 * jnp.log10(jax.vmap(self.cosmology.luminosity_distance)(z_model)) - 5
        ll = self.vll_fn(jnp.stack((z_model, mu_model), axis=1), self.ll_fn_args)
        ll = jnp.sum(jax.scipy.special.logsumexp(ll, axis=0))

        logP_S_C = self.n_sn * self.logP_S_C()

        return ll - logP_S_C

    @ck.forward
    def neg_log_likelihood(self):
        return -self.log_likelihood()


class SpartanLikelihood_GaussianMu(ck.Module):
    def __init__(
        self,
        cosmology: cg.Cosmology,
        spec_z: jnp.ndarray,  # observed redshifts of the supernovae
        mu_mean: jnp.ndarray,  # mean of the observed distance modulus per SN
        mu_var: jnp.ndarray,  # variance of the observed distance modulus per SN
        M0: float,  # Avg absolute magnitude of the supernovae
        model_M_scat: float,  # scatter of the true absolute magnitude of the sim SN
        fiducial_M_scat: float,  # fiducial scatter of the true absolute magnitude of the sim SN
        fiducial_true_M_abs: jnp.ndarray,  # fiducial true absolute magnitude of the sim SN
        fiducial_true_z: jnp.ndarray,  # fiducial true redshift of the sim SN
        fiducial_true_mu: jnp.ndarray,  # fiducial true distance modulus of the sim SN
    ):
        super().__init__()
        self.cosmology = cosmology
        self.spec_z = spec_z
        self.mu_mean = mu_mean
        self.mu_var = mu_var

        # Selection function quantities
        self.model_M_scat = model_M_scat
        self.fiducial_true_z = fiducial_true_z
        self.fiducial_true_mu = fiducial_true_mu
        self.import_denom = -0.5 * (fiducial_true_M_abs - M0) ** 2 / fiducial_M_scat**2

    @abstractmethod
    def logP_S_C(self):
        """Return the log marginal probability of the selection given cosmology."""
        mu_model = (
            5 * jnp.log10(jax.vmap(self.cosmology.luminosity_distance)(self.fiducial_true_z)) - 5
        )
        import_numer = -0.5 * (self.fiducial_true_mu - mu_model) ** 2 / self.model_M_scat**2
        logP_S_C = jax.scipy.special.logsumexp(import_numer - self.import_denom) - jnp.log(
            mu_model.shape[0]
        )
        return logP_S_C

    @ck.forward
    def log_likelihood(self):
        mu_model = 5 * jnp.log10(jax.vmap(self.cosmology.luminosity_distance)(self.spec_z)) - 5
        ll = -0.5 * jnp.sum(((self.mu_mean - mu_model) ** 2) / self.mu_var)

        logP_S_C = self.spec_z.shape[0] * self.logP_S_C()

        return ll - logP_S_C

    @ck.forward
    def neg_log_likelihood(self):
        return -self.log_likelihood()
