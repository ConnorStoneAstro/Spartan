import jax.numpy as jnp
import jax


def gaussian_ll_fn(zmu, args):
    """Gaussian log likelihood function, not including covariance.
    No normalization for the variance is included, as it is not needed for the
    likelihood ratio.

    Args:
        zmu: redshift and distance modulus vector
        args: list containing mean and variance vectors

    Returns:
        log_likelihood: log likelihood value
    """
    mean = args[0]
    var = args[1]
    return -0.5 * jnp.sum((zmu - mean) ** 2 / var)


def gaussian_cov_ll_fn(zmu, args):
    """Gaussian log likelihood function with covariance matrix. No normalization for
    the covariance is included, as it is not needed for the likelihood ratio.

    Args:
        zmu: redshift and distance modulus vector
        args: list containing mean vector and covariance matrix

    Returns:
        log_likelihood: log likelihood value
    """
    mean = args[0]
    inv_cov = args[1]
    diff = zmu - mean
    return -0.5 * (jnp.dot(diff.T, jnp.dot(inv_cov, diff)))


def kde_ll_fn(zmu, args):
    """KDE log likelihood function, not including normalization.
    No normalization for the variance is included, as it is not needed for the
    likelihood ratio.

    Args:
        zmu: redshift and distance modulus vector
        args: list containing KDE object

    Returns:
        log_likelihood: log likelihood value
    """
    zmu_samples = args[0]  # (N, 2) array of samples from the KDE
    kde_sigma = args[1]  # standard deviation of the KDE
    # Compute the KDE log likelihood
    diff = zmu - zmu_samples
    exponent = -0.5 * jnp.sum((diff / kde_sigma) ** 2, axis=1)
    return jax.scipy.special.logsumexp(exponent)
