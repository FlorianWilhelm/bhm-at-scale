from typing import Callable, Optional, Dict

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
from jax.numpy import DeviceArray

# Type declaration
Model = Callable[[DeviceArray], DeviceArray]
Guide = Callable[[DeviceArray], None]


class Plate:
    features = "plate_features"
    stores = "plate_stores"
    days = "plate_days"


class Site:
    disp_param_mu = "disp_param_mu"
    disp_param_sigma = "disp_param_sigma"
    disp_params = "disp_params"
    coef_mus = "coef_mus"
    coef_sigmas = "coef_sigmas"
    coefs = "coefs"
    days = "days"


class Param:
    loc_disp_param_mu = "loc_disp_param_mu"
    scale_disp_param_mu = "scale_disp_param_mu"
    loc_disp_param_logsigma = "loc_disp_param_logsigma"
    scale_disp_param_logsigma = "scale_disp_param_logsigma"
    loc_disp_params = "loc_disp_params"
    scale_disp_params = "scale_disp_params"
    loc_coef_mus = "loc_coef_mus"
    scale_coef_mus = "scale_coef_mus"
    loc_coef_logsigmas = "loc_coef_logsigmas"
    scale_coef_logsigmas = "scale_coef_logsigmas"
    loc_coefs = "loc_coefs"
    scale_coefs = "scale_coefs"


class Features:
    DayOfWeek_1 = 0
    DayOfWeek_2 = 1
    DayOfWeek_3 = 2
    DayOfWeek_4 = 3
    DayOfWeek_5 = 4
    DayOfWeek_6 = 5
    DayOfWeek_7 = 6
    Promo = 7
    StateHoliday_0 = 8
    StateHoliday_1 = 9
    StateHoliday_2 = 10
    StateHoliday_3 = 11
    SchoolHoliday = 12
    Promo2 = 13
    StoreVariant_11 = 14
    StoreVariant_13 = 15
    StoreVariant_21 = 16
    StoreVariant_22 = 17
    StoreVariant_23 = 18
    StoreVariant_31 = 19
    StoreVariant_33 = 20
    StoreVariant_41 = 21
    StoreVariant_43 = 22


def model(X: DeviceArray) -> DeviceArray:
    """Gamma-Poisson hierarchical model for daily sales forecasting

    Args:
        X: input data

    Returns:
        output data
    """
    n_stores, n_days, n_features = X.shape
    n_features -= 1  # remove one dim for target
    eps = 1e-12  # epsilon

    plate_features = numpyro.plate(Plate.features, n_features, dim=-1)
    plate_stores = numpyro.plate(Plate.stores, n_stores, dim=-2)
    plate_days = numpyro.plate(Plate.days, n_days, dim=-1)

    disp_param_mu = numpyro.sample(Site.disp_param_mu, dist.Normal(loc=4.0, scale=1.0))
    disp_param_sigma = numpyro.sample(Site.disp_param_sigma, dist.HalfNormal(scale=1.0))

    with plate_stores:
        with numpyro.handlers.reparam(config={Site.disp_params: TransformReparam()}):
            disp_params = numpyro.sample(
                Site.disp_params,
                dist.TransformedDistribution(
                    dist.Normal(loc=jnp.zeros((n_stores, 1)), scale=0.1),
                    dist.transforms.AffineTransform(disp_param_mu, disp_param_sigma),
                ),
            )

    with plate_features:
        coef_mus = numpyro.sample(
            Site.coef_mus,
            dist.Normal(loc=jnp.zeros(n_features), scale=jnp.ones(n_features)),
        )
        coef_sigmas = numpyro.sample(
            Site.coef_sigmas, dist.HalfNormal(scale=2.0 * jnp.ones(n_features))
        )

        with plate_stores:
            with numpyro.handlers.reparam(config={Site.coefs: TransformReparam()}):
                coefs = numpyro.sample(
                    Site.coefs,
                    dist.TransformedDistribution(
                        dist.Normal(loc=jnp.zeros((n_stores, n_features)), scale=1.0),
                        dist.transforms.AffineTransform(coef_mus, coef_sigmas),
                    ),
                )

    with plate_days, plate_stores:
        targets = X[..., -1]
        features = jnp.nan_to_num(X[..., :-1])  # padded features to 0
        is_observed = jnp.where(
            jnp.isnan(targets), jnp.zeros_like(targets), jnp.ones_like(targets)
        )
        not_observed = 1 - is_observed
        means = (
            is_observed
            * jnp.exp(jnp.sum(jnp.expand_dims(coefs, axis=1) * features, axis=2))
            + not_observed * eps
        )

        betas = is_observed * jnp.exp(-disp_params) + not_observed
        alphas = means * betas
        return numpyro.sample(
            Site.days, dist.GammaPoisson(alphas, betas), obs=jnp.nan_to_num(targets)
        )


def predictive_model(model_params: Dict[str, DeviceArray]) -> Model:
    """This guide determines the parameter when the global and local parameters are already determined

    This can be thought as the `predict` in classical ML

    Args:
        model_params: dict of model parameters

    Returns:
        actual guide function
    """

    def model(X: DeviceArray):
        n_stores, n_days, n_features = X.shape
        n_features -= 1  # remove one dim for target

        plate_features = numpyro.plate(Plate.features, n_features, dim=-1)
        plate_stores = numpyro.plate(Plate.stores, n_stores, dim=-2)
        plate_days = numpyro.plate(Plate.days, n_days, dim=-1)

        disp_param_mu = numpyro.sample(
            Site.disp_param_mu,
            dist.Normal(
                loc=model_params[Param.loc_disp_param_mu],
                scale=model_params[Param.scale_disp_param_mu],
            ),
        )
        disp_param_sigma = numpyro.sample(
            Site.disp_param_sigma,
            dist.TransformedDistribution(
                dist.Normal(
                    loc=model_params[Param.loc_disp_param_logsigma],
                    scale=model_params[Param.scale_disp_param_logsigma],
                ),
                transforms=dist.transforms.ExpTransform(),
            ),
        )

        with plate_stores:
            with numpyro.handlers.reparam(
                config={Site.disp_params: TransformReparam()}
            ):
                disp_params = numpyro.sample(
                    Site.disp_params,
                    dist.TransformedDistribution(
                        dist.Normal(
                            loc=model_params[Param.loc_disp_params],
                            scale=model_params[Param.scale_disp_params],
                        ),
                        dist.transforms.AffineTransform(
                            disp_param_mu, disp_param_sigma
                        ),
                    ),
                )

        with plate_features:
            coef_mus = numpyro.sample(
                Site.coef_mus,
                dist.Normal(
                    loc=model_params[Param.loc_coef_mus],
                    scale=model_params[Param.scale_coef_mus],
                ),
            )
            coef_sigmas = numpyro.sample(
                Site.coef_sigmas,
                dist.TransformedDistribution(
                    dist.Normal(
                        loc=model_params[Param.loc_coef_logsigmas],
                        scale=model_params[Param.scale_coef_logsigmas],
                    ),
                    transforms=dist.transforms.ExpTransform(),
                ),
            )

            with plate_stores:
                with numpyro.handlers.reparam(config={Site.coefs: TransformReparam()}):
                    coefs = numpyro.sample(
                        Site.coefs,
                        dist.TransformedDistribution(
                            dist.Normal(
                                loc=model_params[Param.loc_coefs],
                                scale=model_params[Param.scale_coefs],
                            ),
                            dist.transforms.AffineTransform(coef_mus, coef_sigmas),
                        ),
                    )

        with plate_days, plate_stores:
            features = jnp.nan_to_num(X[..., :-1])
            means = jnp.exp(jnp.sum(jnp.expand_dims(coefs, axis=1) * features, axis=2))
            betas = jnp.exp(-disp_params)
            alphas = means * betas
            return numpyro.sample(Site.days, dist.GammaPoisson(alphas, betas))

    return model


def guide(X: DeviceArray):
    """Guide with parameters of the posterior

    Args:
        X: input data
    """
    n_stores, n_days, n_features = X.shape
    n_features -= 1  # remove one dim for target

    plate_features = numpyro.plate(Plate.features, n_features, dim=-1)
    plate_stores = numpyro.plate(Plate.stores, n_stores, dim=-2)

    disp_param_mu = numpyro.sample(
        Site.disp_param_mu,
        dist.Normal(
            loc=numpyro.param(Param.loc_disp_param_mu, 4.0 * jnp.ones(1)),
            scale=numpyro.param(
                Param.scale_disp_param_mu,
                1.0 * jnp.ones(1),
                constraint=dist.constraints.positive,
            ),
        ),
    )
    disp_param_sigma = numpyro.sample(
        Site.disp_param_sigma,
        dist.TransformedDistribution(
            dist.Normal(
                loc=numpyro.param(Param.loc_disp_param_logsigma, 1.0 * jnp.ones(1)),
                scale=numpyro.param(
                    Param.scale_disp_param_logsigma,
                    0.1 * jnp.ones(1),
                    constraint=dist.constraints.positive,
                ),
            ),
            transforms=dist.transforms.ExpTransform(),
        ),
    )

    with plate_stores:
        with numpyro.handlers.reparam(config={Site.disp_params: TransformReparam()}):
            numpyro.sample(
                Site.disp_params,
                dist.TransformedDistribution(
                    dist.Normal(
                        loc=numpyro.param(
                            Param.loc_disp_params, jnp.zeros((n_stores, 1))
                        ),
                        scale=numpyro.param(
                            Param.scale_disp_params,
                            0.1 * jnp.ones((n_stores, 1)),
                            constraint=dist.constraints.positive,
                        ),
                    ),
                    dist.transforms.AffineTransform(disp_param_mu, disp_param_sigma),
                ),
            )

    with plate_features:
        coef_mus = numpyro.sample(
            Site.coef_mus,
            dist.Normal(
                loc=numpyro.param(Param.loc_coef_mus, jnp.ones(n_features)),
                scale=numpyro.param(
                    Param.scale_coef_mus,
                    0.5 * jnp.ones(n_features),
                    constraint=dist.constraints.positive,
                ),
            ),
        )
        coef_sigmas = numpyro.sample(
            Site.coef_sigmas,
            dist.TransformedDistribution(
                dist.Normal(
                    loc=numpyro.param(Param.loc_coef_logsigmas, jnp.zeros(n_features)),
                    scale=numpyro.param(
                        Param.scale_coef_logsigmas,
                        0.5 * jnp.ones(n_features),
                        constraint=dist.constraints.positive,
                    ),
                ),
                transforms=dist.transforms.ExpTransform(),
            ),
        )

        with plate_stores:
            with numpyro.handlers.reparam(config={Site.coefs: TransformReparam()}):
                numpyro.sample(
                    Site.coefs,
                    dist.TransformedDistribution(
                        dist.Normal(
                            loc=numpyro.param(
                                Param.loc_coefs, jnp.zeros((n_stores, n_features))
                            ),
                            scale=numpyro.param(
                                Param.scale_coefs,
                                0.5 * jnp.ones((n_stores, n_features)),
                                constraint=dist.constraints.positive,
                            ),
                        ),
                        dist.transforms.AffineTransform(coef_mus, coef_sigmas),
                    ),
                )


def local_guide(model_params: Dict[str, DeviceArray]) -> Guide:
    """This guide determines the parameter when the global parameters are already determined

    This is needed when new stores with only a few data points need to be predicted.

    Args:
        model_params: dict of model parameters

    Returns:
        actual guide function
    """

    def guide(X: DeviceArray):
        n_stores, n_days, n_features = X.shape
        n_features -= 1  # remove one dim for target

        plate_features = numpyro.plate(Plate.features, n_features, dim=-1)
        plate_stores = numpyro.plate(Plate.stores, n_stores, dim=-2)

        disp_param_mu = numpyro.sample(
            Site.disp_param_mu,
            dist.Normal(
                loc=model_params[Param.loc_disp_param_mu],
                scale=model_params[Param.scale_disp_param_mu],
            ),
        )

        disp_param_sigma = numpyro.sample(
            Site.disp_param_sigma,
            dist.TransformedDistribution(
                dist.Normal(
                    loc=model_params[Param.loc_disp_param_logsigma],
                    scale=model_params[Param.scale_disp_param_logsigma],
                ),
                transforms=dist.transforms.ExpTransform(),
            ),
        )

        with plate_stores:
            with numpyro.handlers.reparam(
                config={Site.disp_params: TransformReparam()}
            ):
                numpyro.sample(
                    Site.disp_params,
                    dist.TransformedDistribution(
                        dist.Normal(
                            loc=numpyro.param(
                                Param.loc_disp_params, jnp.zeros((n_stores, 1))
                            ),
                            scale=numpyro.param(
                                Param.scale_disp_params,
                                0.1 * jnp.ones((n_stores, 1)),
                                constraint=dist.constraints.positive,
                            ),
                        ),
                        dist.transforms.AffineTransform(
                            disp_param_mu, disp_param_sigma
                        ),
                    ),
                )

        with plate_features:
            coef_mus = numpyro.sample(
                Site.coef_mus,
                dist.Normal(
                    loc=model_params[Param.loc_coef_mus],
                    scale=model_params[Param.scale_coef_mus],
                ),
            )
            coef_sigmas = numpyro.sample(
                Site.coef_sigmas,
                dist.TransformedDistribution(
                    dist.Normal(
                        loc=model_params[Param.loc_coef_logsigmas],
                        scale=model_params[Param.scale_coef_logsigmas],
                    ),
                    transforms=dist.transforms.ExpTransform(),
                ),
            )

            with plate_stores:
                with numpyro.handlers.reparam(config={Site.coefs: TransformReparam()}):
                    numpyro.sample(
                        Site.coefs,
                        dist.TransformedDistribution(
                            dist.Normal(
                                loc=numpyro.param(
                                    Param.loc_coefs, jnp.zeros((n_stores, n_features))
                                ),
                                scale=numpyro.param(
                                    Param.scale_coefs,
                                    0.5 * jnp.ones((n_stores, n_features)),
                                    constraint=dist.constraints.positive,
                                ),
                            ),
                            dist.transforms.AffineTransform(coef_mus, coef_sigmas),
                        ),
                    )

    return guide


def check_model_guide(
    X: DeviceArray, *, model: Optional[Model] = None, guide: Optional[Guide] = None
):
    """Do some really simple checks to see that model and guide work at least syntactically

    Args:
        X: array of data that suits the model
        model: some model function
        guide: some guide function

    Returns:
        trace if a model was supplied otherwise None
    """
    assert (
        model is not None or guide is not None
    ), "At least a model or a guide need to be provided!"
    rng_key = random.PRNGKey(0)
    if guide is not None:
        # prime and check if guide swallows data
        numpyro.handlers.seed(guide, rng_key)(X)
    if model is not None:
        primed_model = numpyro.handlers.seed(model, rng_key)
        primed_model(X)  # test if primed model swallows data
        numpyro.handlers.trace(primed_model).get_trace(X)  # calculate trace
