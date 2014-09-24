import warnings

from .general import is_iterable
import theano
# Delay import of pylearn2.config.yaml_parse and pylearn2.datasets.control
# to avoid circular imports
yaml_parse = None
control = None
from itertools import izip
cuda = None

import numpy as np

from functools import partial
WRAPPER_ASSIGNMENTS = ('__module__', '__name__')
WRAPPER_CONCATENATIONS = ('__doc__',)
WRAPPER_UPDATES = ('__dict__',)


def make_name(variable, anon="anonymous_variable"):
    """
    If variable has a name, returns that name. Otherwise, returns anon.

    Parameters
    ----------
    variable : tensor_like
        WRITEME

    Returns
    -------
    WRITEME
    """

    if hasattr(variable, 'name') and variable.name is not None:
        return variable.name

    return anon


def sharedX(value, name=None, borrow=False):
    """
    Transform value into a shared variable of type floatX

    Parameters
    ----------
    value : WRITEME
    name : WRITEME
    borrow : WRITEME

    Returns
    -------
    WRITEME
    """

    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow)


def as_floatX(variable):
    """
    Casts a given variable into dtype `config.floatX`. Numpy ndarrays will
    remain numpy ndarrays, python floats will become 0-D ndarrays and
    all other types will be treated as theano tensors

    Parameters
    ----------
    variable : WRITEME

    Returns
    -------
    WRITEME
    """

    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)


def constantX(value):
    """
    Returns a constant of value `value` with floatX dtype

    Parameters
    ----------
    variable : WRITEME

    Returns
    -------
    WRITEME
    """
    return theano.tensor.constant(np.asarray(value,
                                             dtype=theano.config.floatX))


def subdict(d, keys):
    """
    Create a subdictionary of d with the keys in keys

    Parameters
    ----------
    d : WRITEME
    keys : WRITEME

    Returns
    -------
    WRITEME
    """
    result = {}
    for key in keys:
        if key in d:
            result[key] = d[key]
    return result


def safe_update(dict_to, dict_from):
    """
    Like dict_to.update(dict_from), except don't overwrite any keys.

    Parameters
    ----------
    dict_to : WRITEME
    dict_from : WRITEME

    Returns
    -------
    WRITEME
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


class CallbackOp(theano.gof.Op):
    """
    A Theano Op that implements the identity transform but also does an
    arbitrary (user-specified) side effect.

    Parameters
    ----------
    callback : WRITEME
    """
    view_map = {0: [0]}

    def __init__(self, callback):
        self.callback = callback

    def make_node(self, xin):
        """
        .. todo::

            WRITEME
        """
        xout = xin.type.make_variable()
        return theano.gof.Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        """
        .. todo::

            WRITEME
        """
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        self.callback(xin)

    def grad(self, inputs, output_gradients):
        """
        .. todo::

            WRITEME
        """
        return output_gradients

    def R_op(self, inputs, eval_points):
        """
        .. todo::

            WRITEME
        """
        return [x for x in eval_points]

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return type(self) == type(other) and self.callback == other.callback

    def hash(self):
        """
        .. todo::

            WRITEME
        """
        return hash(self.callback)


def get_dataless_dataset(model):
    """
    Loads the dataset that model was trained on, without loading data.
    This is useful if you just need the dataset's metadata, like for
    formatting views of the model's weights.

    Parameters
    ----------
    model : Model

    Returns
    -------
    dataset : Dataset
        The data-less dataset as described above.
    """

    global yaml_parse
    global control

    if yaml_parse is None:
        from pylearn2.config import yaml_parse

    if control is None:
        from pylearn2.datasets import control

    control.push_load_data(False)
    try:
        rval = yaml_parse.load(model.dataset_yaml_src)
    finally:
        control.pop_load_data()
    return rval


def safe_zip(*args):
    """
    Like zip, but ensures arguments are of same length
    """
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)


def safe_izip(*args):
    """
    Like izip, but ensures arguments are of same length
    """
    assert all([len(arg) == len(args[0]) for arg in args])
    return izip(*args)


def gpu_mem_free():
    """
    Returns
    -------
    megs_free : float
        Number of megabytes of memory free on the GPU used by Theano
    """
    global cuda
    if cuda is None:
        from theano.sandbox import cuda
    return cuda.mem_info()[0]/1024./1024


class _ElemwiseNoGradient(theano.tensor.Elemwise):
    """
    A Theano Op that applies an elementwise transformation and reports
    having no gradient.
    """
    def connection_pattern(self, node):
        """
        Report being disconnected to all inputs in order to have no gradient
        at all.
        """
        return [[False]]

    def grad(self, inputs, output_gradients):
        """
        Report being disconnected to all inputs in order to have no gradient
        at all.
        """
        return [theano.gradient.DisconnectedType()()]

# Call this on a theano variable to make a copy of that variable
# No gradient passes through the copying operation
# This is equivalent to making my_copy = var.copy() and passing
# my_copy in as part of consider_constant to tensor.grad
# However, this version doesn't require as much long range
# communication between parts of the code
block_gradient = _ElemwiseNoGradient(theano.scalar.identity)

def is_block_gradient(op):
    """
    Parameters
    ----------
    op : object

    Returns
    -------
    is_block_gradient: bool
        True if op is a gradient-blocking op, False otherwise
    """

    return isinstance(op, _ElemwiseNoGradient)


def safe_union(a, b):
    """
    Does the logic of a union operation without the non-deterministic ordering
    of python sets.

    Parameters
    ----------
    a : list
    b : list

    Returns
    -------
    c : list
        A list containing one copy of each element that appear in at least one
        of `a` or `b`.
    """
    if not isinstance(a, list):
        raise TypeError("Expected first argument to be a list, but got " +
                        str(type(a)))
    assert isinstance(b, list)
    c = []
    for x in a + b:
        if x not in c:
            c.append(x)
    return c

# This was moved to theano, but I include a link to avoid breaking
# old imports
from theano.printing import hex_digest as _hex_digest
def hex_digest(*args, **kwargs):
    warnings.warn("hex_digest has been moved into Theano. "
            "pylearn2.utils.hex_digest will be removed on or after "
            "2014-08-26")

def function(*args, **kwargs):
    """
    A wrapper around theano.function that disables the on_unused_input error.
    Almost no part of pylearn2 can assume that an unused input is an error, so
    the default from theano is inappropriate for this project.
    """
    return theano.function(*args, on_unused_input='ignore', **kwargs)


def grad(*args, **kwargs):
    """
    A wrapper around theano.gradient.grad that disable the disconnected_inputs
    error. Almost no part of pylearn2 can assume that a disconnected input
    is an error.
    """
    return theano.gradient.grad(*args, disconnected_inputs='ignore', **kwargs)


# Groups of Python types that are often used together in `isinstance`
py_integer_types = (int, long, np.integer)
py_float_types = (float, np.floating)
py_complex_types = (complex, np.complex)
py_number_types = (int, long, float, complex, np.number)


def get_choice(choice_to_explanation):
    """
    WRITEME

    Parameters
    ----------
    choice_to_explanation : dict
        Dictionary mapping possible user responses to strings describing what \
        that response will cause the script to do

    Returns
    -------
    WRITEME
    """
    d = choice_to_explanation

    for key in d:
        print '\t'+key + ': '+d[key]
    prompt = '/'.join(d.keys())+'? '

    first = True
    choice = ''
    while first or choice not in d.keys():
        if not first:
            print 'unrecognized choice'
        first = False
        choice = raw_input(prompt)
    return choice


def float32_floatX(f):
    """
    This function changes floatX to float32 for the call to f. Useful in GPU
    tests.

    Parameters
    ----------
    f : WRITEME

    Returns
    -------
    WRITEME
    """
    def new_f(*args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        old_floatX = theano.config.floatX
        theano.config.floatX = 'float32'
        try:
            f(*args, **kwargs)
        finally:
            theano.config.floatX = old_floatX

    # If we don't do that, tests function won't be run.
    new_f.func_name = f.func_name
    return new_f


def update_wrapper(wrapper,
                   wrapped,
                   assigned=WRAPPER_ASSIGNMENTS,
                   concatenated=WRAPPER_CONCATENATIONS,
                   append=False,
                   updated=WRAPPER_UPDATES):
    """
    A Python decorator which acts like `functools.update_wrapper` but also has
    the ability to concatenate attributes.

    Parameters
    ----------
    wrapper : functon
        Function to be updated
    wrapped : function
        Original function
    assigned : tuple, optional
        Tuple naming the attributes assigned directly from the wrapped function
        to the wrapper function. Defaults to `utils.WRAPPER_ASSIGNMENTS`.
    concatenated : tuple, optional
        Tuple naming the attributes from the wrapped function concatenated with
        the ones from the wrapper function. Defaults to
        `utils.WRAPPER_CONCATENATIONS`.
    append : bool, optional
        If True, appends wrapped attributes to wrapper attributes instead of
        prepending them. Defaults to False.
    updated : tuple, optional
        Tuple naming the attributes of the wrapper that are updated with the
        corresponding attribute from the wrapped function. Defaults to
        `functools.WRAPPER_UPDATES`.

    Returns
    -------
    wrapper : function
        Updated wrapper function

    Notes
    -----
    This can be used to concatenate the wrapper's docstring with the wrapped's
    docstring and should help reduce the ammount of documentation to write: one
    can use this decorator on child classes' functions when their
    implementation is similar to the one of the parent class. Conversely, if a
    function defined in a child class departs from its parent's implementation,
    one can simply explain the differences in a 'Notes' section without
    re-writing the whole docstring.
    """
    for attr in assigned:
        setattr(wrapper, attr, getattr(wrapped, attr))
    for attr in concatenated:
        # Make sure attributes are not None
        if getattr(wrapped, attr) is None:
            setattr(wrapped, attr, "")
        if getattr(wrapper, attr) is None:
            setattr(wrapper, attr, "")
        if append:
            setattr(wrapper,
                    attr,
                    getattr(wrapped, attr) + getattr(wrapper, attr))
        else:
            setattr(wrapper,
                    attr,
                    getattr(wrapper, attr) + getattr(wrapped, attr))
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper


def wraps(wrapped,
          assigned=WRAPPER_ASSIGNMENTS,
          concatenated=WRAPPER_CONCATENATIONS,
          append=False,
          updated=WRAPPER_UPDATES):
    """
    Decorator factory to apply `update_wrapper()` to a wrapper function

    Returns a decorator that invokes `update_wrapper()` with the decorated
    function as the wrapper argument and the arguments to `wraps()` as the
    remaining arguments. Default arguments are as for `update_wrapper()`.
    This is a convenience function to simplify applying `partial()` to
    `update_wrapper()`.

    Examples
    --------
    >>> class Parent(object):
    ...     def f(x):
    ...        '''
    ...        Adds 1 to x
    ...
    ...        Parameters
    ...        ----------
    ...        x : int
    ...            Variable to increment by 1
    ...
    ...        Returns
    ...        -------
    ...        rval : int
    ...            x incremented by 1
    ...        '''
    ...        rval = x + 1
    ...        return rval
    ...
    >>> class Child(Parent):
    ...     @wraps(Parent.f)
    ...     def f(x):
    ...        '''
    ...        Notes
    ...        -----
    ...        Also prints the incremented value
    ...        '''
    ...        rval = x + 1
    ...        print rval
    ...        return rval
    ...
    >>> c = Child()
    >>> print c.f.__doc__

        Adds 1 to x

        Parameters
        ----------
        x : int
            Variable to increment by 1

        Returns
        -------
        rval : int
           x incremented by 1

        Notes
        -----
        Also prints the incremented value
    """
    return partial(update_wrapper, wrapped=wrapped, assigned=assigned,
                   append=append,updated=updated)
