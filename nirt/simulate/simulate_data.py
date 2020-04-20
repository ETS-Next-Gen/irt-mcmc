"""Item response function (computed IRFs and parametric models) plots."""
import numpy as np
import numpy.matlib
from scipy.stats import invgamma


def generate_dichotomous_responses(num_persons, num_items, num_latent_dimensions, asymptote=0.25, discrimination=1):
    """
    Generates simulated parametric IRT dichotomous item response data. Generates binary item responses. Assumes person
    abilities theta[:,c], c=1,...,C follows a N(0, InvGamma(1,1)) distribution. Item difficulties are uniformly spaced.
    Item discrimination is uniformly random in [0.5,1.5]. Item classification is defined so that there are roughly equal
    number of items per class.

    Args:
        num_persons: int, number of persons.
        num_items: int, number of items.
        num_latent_dimensions: int, number of item classes = number of latest ability dimensions.
        discrimination: item discrimination factor.
        asymptote: float, probability of guessing = asymptotic IRF value at theta -> -infinity.

    Returns:
        x: array<float>, shape=(num_persons, num_items) binary item response matrix.
        theta: array<float>, shape=(P, C) latent person ability matrix.
        c: (I,): array<int> item classification (0 <= c[i] < C).
    """
    # Generate latent ability distribution variances.
    alpha_theta, beta_theta = 1, 1
    rv = invgamma(a=alpha_theta, scale=beta_theta)
    v = rv.rvs(num_latent_dimensions)

    # Generate normally distributed student latent abilities. theta_c ~ N(0, invgamma(a_c,b_c))
    cov = np.diag(v)
    theta = np.random.multivariate_normal(np.zeros((num_latent_dimensions,)), cov, num_persons)

    # Generate item difficulty parameters.
    # Discrimination is uniform[0.5, 1.5].
    if discrimination is None:
        a = np.random.random(size=(num_items,)) + 0.5
    else:
        a = np.ones((num_items,)) * discrimination
    # Difficulty is equally spaced from -3 to 3.
    b = np.linspace(-3, 3, num=num_items)
    # Item i measures sub-scale c[i]. Select about the same number of items per subscale,
    # then randomly permute the item order.
    c = np.random.permutation(np.matlib.repmat(np.arange(num_latent_dimensions, dtype=int),
                                               int(np.ceil(num_items / num_latent_dimensions)), 1).ravel()[:num_items])

    # Generate item responses (the observed data) (3PL model).
    t = np.exp(a*(theta[:, c] - b))
    p_correct = asymptote + (1 - asymptote) * (t / (1 + t))
    x = np.random.binomial(1, p=p_correct)
    return x, theta, b, c


def generate_response_times(num_persons, num_items):
    """
    Generates simulated item response times. This is a log-normal model.

    Citation: Sandip Sinharay, S. and van Rijn, P., Assessing Fit of the Lognormal Model for Response Times, JEBS,
    to appear (2020).

    Args:
        num_persons: int, number of persons.
        num_items: int, number of items.

    Returns:
        t: array<float>, shape=(num_persons, num_items) item response time matrix.
    """
    alpha = np.random.normal(1.87, 0.15, num_items)
    beta = np.random.normal(4, 0.45, num_items)
    tau = np.random.normal(0, 0.3, num_persons)

    return np.exp(np.random.normal(beta[None, :] - tau[:, None], 1/alpha[None, :] ** 2, size=(num_persons, num_items)))
