from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn import compilable
from ngclearn import Compartment


class AdaptivePrecisionErrorCell(JaxComponent):
    """
    A precision-weighted Gaussian error cell -- a drop-in extension of
    ngclearn's GaussianErrorCell with the exact same compartment interface
    (mu, target, dmu, dtarget, Sigma, dSigma, modulator, mask, L), so it can
    replace GaussianErrorCell anywhere in the wiring graph with no other
    changes needed.

    Background
    ----------
    GaussianErrorCell already computes `dmu = (target - mu) / Sigma` -- that
    division by Sigma IS precision weighting (precision = 1/Sigma, in the
    Friston/free-energy sense). But in the base cell, Sigma is a fixed
    constant (default 1.0) for every unit and is never adapted -- the code
    comments literally say "no derivative is calculated at this time for
    sigma". So every layer currently gets equal-confidence treatment.

    This cell fills that gap: each unit maintains a running (EMA) estimate
    of its own squared prediction error, and uses that as its estimated
    variance/Sigma. Units/features that have historically had large,
    persistent errors get a large Sigma -> small precision -> down-weighted
    dmu/dtarget (the network "gives up" trying to hard-fit an unreliable
    prediction). Units with small, well-explained errors get a small Sigma
    -> high precision -> their error signal dominates the update, exactly
    like a confident sensory channel dominating a noisy one in hierarchical
    predictive coding / Kalman-filter-style precision weighting.

    Args:
        name: string name of this cell
        n_units: number of cellular entities (population size)
        batch_size: batch size dimension (Default: 1)
        sigma_init: initial variance estimate for every unit
        sigma_min: floor on the variance estimate, to keep precision (1/Sigma)
                   from blowing up when a unit's error is near zero
        momentum: EMA decay rate for the running variance estimate (close to
                  1 = slow/stable precision estimate, closer to biological
                  timescales; lower = precision reacts faster but noisier)
    """

    def __init__(self, name, n_units, batch_size=1, sigma_init=1., sigma_min=0.1,
                 momentum=0.95, shape=None, **kwargs):
        super().__init__(name, **kwargs)

        _shape = (batch_size, n_units)
        if shape is None:
            shape = (n_units,)
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])
        self.shape = shape
        self.n_units = n_units
        self.batch_size = batch_size
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.momentum = momentum

        restVals = jnp.zeros(_shape)
        self.L = Compartment(0., display_name="Gaussian Log likelihood", units="nats")
        self.mu = Compartment(restVals, display_name="Gaussian mean")
        self.dmu = Compartment(restVals)
        # Sigma is now per-unit (n_units,) and adaptive, not a fixed global scalar
        self.Sigma = Compartment(jnp.ones((n_units,)) * sigma_init,
                                  display_name="Adaptive per-unit precision (1/Sigma)")
        self.dSigma = Compartment(jnp.zeros((n_units,)))
        self.target = Compartment(restVals, display_name="Gaussian data/target variable")
        self.dtarget = Compartment(restVals)
        self.modulator = Compartment(restVals + 1.0)
        self.mask = Compartment(restVals + 1.0)
        # persistent running estimate of each unit's error variance
        self.running_var = Compartment(jnp.ones((n_units,)) * sigma_init)

    @compilable
    def advance_state(self, dt):
        mu = self.mu.get()
        target = self.target.get()
        running_var = self.running_var.get()
        modulator = self.modulator.get()
        mask = self.mask.get()

        _dmu = (target - mu)  # raw error/mismatch, same as base cell

        # per-unit current-batch squared error (the "instantaneous" evidence
        # about how unreliable this unit's prediction is right now)
        current_sq_err = jnp.mean(jnp.square(_dmu), axis=0)

        # slow running estimate of each unit's error variance -- this IS the
        # adaptive Sigma; a unit that's persistently wrong accumulates a large
        # variance estimate here
        new_running_var = self.momentum * running_var + (1. - self.momentum) * current_sq_err
        Sigma = jnp.clip(new_running_var, min=self.sigma_min)

        # precision-weighted error: divide by the *learned*, per-unit Sigma
        # instead of a fixed constant -- this is the actual precision-weighting step
        dmu = _dmu / Sigma
        dtarget = -dmu
        dSigma = Sigma * 0. + 1.
        L = -jnp.sum(jnp.square(_dmu) / Sigma) * 0.5

        dmu = dmu * modulator * mask
        dtarget = dtarget * modulator * mask
        mask = mask * 0. + 1.

        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.dSigma.set(dSigma)
        self.L.set(jnp.squeeze(L))
        self.mask.set(mask)
        self.Sigma.set(Sigma)
        self.running_var.set(new_running_var)

    @compilable
    def reset(self):
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        restVals = jnp.zeros(_shape)

        self.dmu.set(restVals)
        self.dtarget.set(restVals)
        self.dSigma.set(jnp.zeros((self.n_units,)))
        self.target.set(restVals)
        self.mu.set(restVals)
        self.modulator.set(restVals + 1.)
        self.L.set(0.)
        self.mask.set(jnp.ones(_shape))
        self.Sigma.set(jnp.ones((self.n_units,)) * self.sigma_init)
        self.running_var.set(jnp.ones((self.n_units,)) * self.sigma_init)

    @classmethod
    def help(cls):
        properties = {
            "cell_type": "AdaptivePrecisionErrorCell - Gaussian error cell whose "
                         "per-unit Sigma (inverse precision) adapts via a running "
                         "estimate of that unit's own squared error, implementing "
                         "genuine precision-weighted predictive coding error signals."
        }
        compartment_props = {
            "inputs": {"mu": "predicted value(s)", "target": "target/goal value(s)",
                       "modulator": "modulatory scaling signal(s)",
                       "mask": "binary gating mask"},
            "outputs": {"L": "local loss", "dmu": "precision-weighted error w.r.t. mu",
                        "dtarget": "precision-weighted error w.r.t. target",
                        "Sigma": "current learned per-unit variance/inverse-precision"},
        }
        hyperparams = {
            "n_units": "number of units", "batch_size": "batch size",
            "sigma_init": "initial variance estimate",
            "sigma_min": "floor on variance to bound precision",
            "momentum": "EMA decay rate for the running variance estimate",
        }
        return {cls.__name__: properties, "compartments": compartment_props,
                "dynamics": "Gaussian(x=target; mu, Sigma=EMA(error^2))",
                "hyperparameters": hyperparams}




