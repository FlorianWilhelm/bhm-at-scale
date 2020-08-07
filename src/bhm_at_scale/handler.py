import pickle
from typing import IO, Optional, Dict

from jax import random, lax
from jax import jit
from jax.experimental.optimizers import OptimizerState
import jax.numpy as jnp
from jax.numpy import DeviceArray
from numpyro import optim
from numpyro.infer import ELBO, SVI, Predictive
from numpyro.infer.svi import SVIState

from .model import Model, Guide


class ModelHandler(object):
    def __init__(self,
                 model: Model,
                 guide: Guide,
                 rng_key: int = 0,
                 *,
                 loss: ELBO = ELBO(num_particles=1),
                 optim_builder: optim.optimizers.optimizer = optim.Adam):
        """Handling the model and guide for training and prediction

        Args:
            model: function holding the numpyro model
            guide: function holding the numpyro guide
            rng_key: random key as int
            loss: loss to optimize
            optim_builder: builder for an optimizer
        """
        self.model = model
        self.guide = guide
        self.rng_key = random.PRNGKey(rng_key)  # current random key
        self.loss = loss
        self.optim_builder = optim_builder
        self.svi = None
        self.svi_state = None
        self.optim = None
        self.log_func = print  # overwrite e.g. logger.info(...)

    def reset_svi(self):
        """Reset the current SVI state"""
        self.svi = None
        self.svi_state = None
        return self

    def init_svi(self, X: DeviceArray, *, lr: float, **kwargs):
        """Initialize the SVI state

        Args:
            X: input data
            lr: learning rate
            kwargs: other keyword arguments for optimizer
        """
        self.optim = self.optim_builder(lr, **kwargs)
        self.svi = SVI(self.model, self.guide, self.optim, self.loss)
        svi_state = self.svi.init(self.rng_key, X)
        if self.svi_state is None:
            self.svi_state = svi_state
        return self

    @property
    def optim_state(self) -> OptimizerState:
        """Current optimizer state"""
        assert self.svi_state is not None, "'init_svi' needs to be called first"
        return self.svi_state.optim_state

    @optim_state.setter
    def optim_state(self, state: OptimizerState):
        """Set current optimizer state"""
        self.svi_state = SVIState(state, self.rng_key)

    def dump_optim_state(self, fh: IO):
        """Pickle and dump optimizer state to file handle"""
        pickle.dump(optim.optimizers.unpack_optimizer_state(self.optim_state[1]), fh)
        return self

    def load_optim_state(self, fh: IO):
        """Read and unpickle optimizer state from file handle"""
        state = optim.optimizers.pack_optimizer_state(pickle.load(fh))
        iter0 = jnp.array(0)
        self.optim_state = (iter0, state)
        return self

    @property
    def optim_total_steps(self) -> int:
        """Returns the number of performed iterations in total"""
        return int(self.optim_state[0])

    def _fit(self, X: DeviceArray, n_epochs) -> float:
        @jit
        def train_epochs(svi_state, n_epochs):
            def train_one_epoch(_, val):
                loss, svi_state = val
                svi_state, loss = self.svi.update(svi_state, X)
                return loss, svi_state

            return lax.fori_loop(0, n_epochs, train_one_epoch, (0., svi_state))

        loss, self.svi_state = train_epochs(self.svi_state, n_epochs)
        return float(loss / X.shape[0])

    def _log(self, n_digits, epoch, loss):
        msg = f"epoch: {str(epoch).rjust(n_digits)} loss: {loss: 16.4f}"
        self.log_func(msg)

    def fit(self,
            X: DeviceArray,
            *,
            n_epochs: int,
            log_freq: int = 0,
            lr: float,
            **kwargs) -> float:
        """Train but log with a given frequency

        Args:
            X: input data
            n_epochs: total number of epochs
            log_freq: log loss every log_freq number of eppochs
            lr: learning rate
            kwargs: parameters of `init_svi`

        Returns:
            final loss of last epoch
        """
        self.init_svi(X, lr=lr, **kwargs)
        if log_freq <= 0:
            self._fit(X, n_epochs)
        else:
            loss = self.svi.evaluate(self.svi_state, X) / X.shape[0]

            curr_epoch = 0
            n_digits = len(str(abs(n_epochs)))
            self._log(n_digits, curr_epoch, loss)

            for i in range(n_epochs // log_freq):
                curr_epoch += log_freq
                loss = self._fit(X, log_freq)
                self._log(n_digits, curr_epoch, loss)

            rest = n_epochs % log_freq
            if rest > 0:
                curr_epoch += rest

                loss = self._fit(X, rest)
                self._log(n_digits, curr_epoch, loss)

        loss = self.svi.evaluate(self.svi_state, X) / X.shape[0]
        self.rng_key = self.svi_state.rng_key
        return float(loss)

    @property
    def model_params(self) -> Optional[Dict[str, DeviceArray]]:
        """Gets model parameters

        Returns:
            dict of model parameters
        """
        if self.svi is not None:
            return self.svi.get_params(self.svi_state)
        else:
            return None

    def predict(self, X: DeviceArray, **kwargs) -> DeviceArray:
        """Predict the parameters of a model specified by `return_sites`

        Args:
            X: input data
            kwargs: keyword arguments for numpro `Predictive`

        Returns:
            samples for all sample sites
        """
        self.init_svi(X, lr=0.)  # dummy initialization
        predictive = Predictive(self.model,
                                guide=self.guide,
                                params=self.model_params,
                                **kwargs)
        samples = predictive(self.rng_key, X)
        return samples

