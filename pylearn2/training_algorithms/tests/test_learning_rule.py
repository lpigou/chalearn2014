import numpy as np
import warnings

import theano.tensor as T
from theano.tests import disturb_mem

from pylearn2.costs.cost import SumOfCosts
from pylearn2.testing.cost import SumOfOneHalfParamsSquared
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.testing.cost import SumOfParams
from pylearn2.testing.datasets import ArangeDataset
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.training_algorithms.learning_rule import AdaDelta
from pylearn2.utils import sharedX

from test_sgd import DummyCost, DummyModel


def test_momentum():
    """
    Make sure that learning_rule.Momentum obtains the same parameter values as
    with a hand-crafted sgd w/ momentum implementation, given a dummy model and
    learning rate scaler for each parameter.
    """
    # We include a cost other than SumOfParams so that data is actually
    # queried from the training set, and the expected number of updates
    # are applied.
    cost = SumOfCosts([SumOfParams(), (0., DummyCost())])

    scales = [ .01, .02, .05, 1., 5. ]
    shapes = [(1,), (9,), (8, 7), (6, 5, 4), (3, 2, 2, 2)]

    model = DummyModel(shapes, lr_scalers=scales)
    dataset = ArangeDataset(1)
    learning_rate = .001
    momentum = 0.5

    sgd = SGD(cost=cost,
              learning_rate=learning_rate,
              learning_rule = Momentum(momentum),
              batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    manual = [param.get_value() for param in model.get_params()]
    inc = [ - learning_rate * scale for param, scale in
            zip(manual, scales)]
    manual = [param + i for param, i in zip(manual, inc)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value()) for manual_param,
            sgd_param in zip(manual, model.get_params()))

    manual = [param - learning_rate * scale + i * momentum for param, scale, i in
            zip(manual, scales, inc)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value()) for manual_param,
            sgd_param in zip(manual, model.get_params()))


def test_adadelta():
    """
    Make sure that learning_rule.AdaDelta obtains the same parameter values as
    with a hand-crafted AdaDelta implementation, given a dummy model and
    learning rate scaler for each parameter.
    
    Reference:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.
    """

    # We include a cost other than SumOfParams so that data is actually
    # queried from the training set, and the expected number of updates
    # are applied.
    cost = SumOfCosts([SumOfOneHalfParamsSquared(), (0., DummyCost())])

    scales = [ .01, .02, .05, 1., 5. ]
    shapes = [(1,), (9,), (8, 7), (6, 5, 4), (3, 2, 2, 2)]

    model = DummyModel(shapes, lr_scalers=scales)
    dataset = ArangeDataset(1)
    learning_rate = .001
    decay = 0.95

    sgd = SGD(cost=cost,
              learning_rate=learning_rate,
              learning_rule = AdaDelta(decay),
              batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    state = {}
    for param in model.get_params():
        param_shape = param.get_value().shape
        state[param] = {}
        state[param]['g2'] = np.zeros(param_shape)
        state[param]['dx2'] = np.zeros(param_shape)

    def adadelta_manual(model, state):
        inc = []
        rval = []
        for scale, param in zip(scales, model.get_params()):
            pstate = state[param]
            param_val =  param.get_value()
            # begin adadelta
            pstate['g2'] = decay * pstate['g2'] + (1. - decay) * param_val**2
            rms_g_t = np.sqrt(pstate['g2'] + scale * learning_rate)
            rms_dx_tm1 = np.sqrt(pstate['dx2'] + scale * learning_rate)
            dx_t = - rms_dx_tm1 / rms_g_t * param_val
            pstate['dx2'] = decay * pstate['dx2'] + (1. - decay) * dx_t**2
            rval += [param_val + dx_t]
        return rval

    manual = adadelta_manual(model, state)
    sgd.train(dataset=dataset)
    assert all(np.allclose(manual_param, sgd_param.get_value()) for manual_param,
            sgd_param in zip(manual, model.get_params()))

    manual = adadelta_manual(model, state)
    sgd.train(dataset=dataset)
    assert all(np.allclose(manual_param, sgd_param.get_value()) for manual_param,
            sgd_param in zip(manual, model.get_params()))
