__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import numpy as N
from pylearn2.datasets import dense_design_matrix

class Wiskott(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME
    """
    def __init__(self):

        X = 1. - N.load("/data/lisa/data/wiskott/wiskott_fish_layer0_15_standard_64x64_shuffled.npy")


        view_converter = dense_design_matrix.DefaultViewConverter((64,64,1))

        super(Wiskott,self).__init__(X = X, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))
    #
#
