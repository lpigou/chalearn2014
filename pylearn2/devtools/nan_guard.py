"""
Functionality for detecting NaNs in a Theano graph.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from theano.compile import Mode
import theano
import numpy as np
from pylearn2.models.dbm import flatten

class NanGuardMode(Mode):
    """
    A Theano compilation Mode that makes the compiled function automatically
    detect NaNs and Infs and detect an error if they occur.

    Parameters
    ----------
    nan_is_error : bool
        If True, raise an error anytime a NaN is encountered
    inf_is_error: bool
        If True, raise an error anytime an Inf is encountered
    """
    def __init__(self, nan_is_error, inf_is_error):
        def do_check_on(var, nd, f, is_input):
            """
            Checks `var` for NaNs / Infs. If detected, raises an exception
            and / or prints information about `nd`, `f`, and `is_input` to
            help the user determine the cause of the invalid values.

            Parameters
            ----------
            var : numpy.ndarray
                The value to be checked.
            nd : theano.gof.Apply
                The Apply node being executed
            f : callable
                The thunk for the apply node
            is_input : bool
                If True, `var` is an input to `nd`.
                If False, it is an output.
            """
            error = False
            if nan_is_error:
                if np.any(np.isnan(var)):
                    print 'NaN detected'
                    error = True
            if inf_is_error:
                if np.any(np.isinf(var)):
                    print 'Inf detected'
                    error = True
            if np.abs(var).max() > 1e10:
                print 'Big value detected'
                error = True
            if error:
                if is_input:
                    print 'In an input'
                else:
                    print 'In an output'
                print 'Inputs: '
                for ivar, ival in zip(nd.inputs, f.inputs):
                    print 'var'
                    print ivar
                    print theano.printing.min_informative_str(ivar)
                    print 'val'
                    print ival
                print 'Node:'
                print nd
                assert False

        def nan_check(i, node, fn):
            """
            Runs `fn` while checking its inputs and outputs for NaNs / Infs

            Parameters
            ----------
            i : currently ignored (TODO: determine why it is here or remove)
            node : theano.gof.Apply
                The Apply node currently being executed
            fn : callable
                The thunk to execute for this Apply node
            """
            inputs = fn.inputs
            # TODO: figure out why individual inputs are themselves lists sometimes
            for x in flatten(inputs):
                do_check_on(x, node, fn, True)
            fn()
            outputs = fn.outputs
            for j, x in enumerate(flatten(outputs)):
                do_check_on(x, node, fn, False)

        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()], [nan_check])
        super(NanGuardMode, self).__init__(wrap_linker, optimizer='fast_run')
