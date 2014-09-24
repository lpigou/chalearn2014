"""
Plugins for the Train object.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np


class TrainExtension(object):
    """
    An object called by pylearn2.train.Train at various
    points during learning.
    Useful for adding custom features to the basic learning
    procedure.

    This base class implements all callback methods as no-ops.
    To add a feature to the Train class, implement a subclass of this
    base class that overrides any subset of these no-op methods.
    """

    def on_save(self, model, dataset, algorithm):
        """
        Train calls this immediately before it saves the model.

        Parameters
        ----------
        model : object
            The model object being trained (implementing some subset of the \
            `pylearn2.models` interface).

        dataset : object
            The dataset object being trained (implementing the \
            `pylearn2.datasets` interface).

        algorithm : object
            The object representing the training algorithm being \
            used to train the model (and thus implementing the \
            `pylearn2.training_algorithms` interface).
        """

    def on_monitor(self, model, dataset, algorithm):
        """
        Train calls this immediately after each call to the Monitor
        (i.e., when training begins, and at the end of each epoch).

        Parameters
        ----------
        model : object
            The model object being trained (implementing some \
            subset of the `pylearn2.models` interface).

        dataset : object
            The dataset object being trained (implementing the \
            `pylearn2.datasets` interface).

        algorithm : object
            The object representing the training algorithm being \
            used to train the model (and thus implementing the \
            `pylearn2.training_algorithms` interface).
        """

    def setup(self, model, dataset, algorithm):
        """
        Train calls this immediately upon instantiation,
        before any monitoring is done.

        Parameters
        ----------
        model : object
            The model object being trained (implementing some \
            subset of the `pylearn2.models` interface).

        dataset : object
            The dataset object being trained (implementing the \
            `pylearn2.datasets` interface).

        algorithm : object
            The object representing the training algorithm being \
            used to train the model (and thus implementing the \
            `pylearn2.training_algorithms` interface).
        """

class SharedSetter(TrainExtension):
    """
    Sets shared variables to take on the specified values after the
    specified amounts of epochs have taken place.

    epoch_updates = [ [i, x, y] ]

    means run x.set_value(cast(y))

    after i epochs have passed.

    Parameters
    ----------
    epoch_updates : WRITEME
    """

    def __init__(self, epoch_updates):
        self._count = 0
        self._epoch_to_updates = {}
        self._vars = set([])
        for update in epoch_updates:
            epoch, var, val = update
            self._vars.add(var)
            if epoch not in self._epoch_to_updates:
                self._epoch_to_updates[epoch] = []
            assert hasattr(var, 'get_value')
            assert var.name is not None
            self._epoch_to_updates[epoch].append((var,val))

    def on_monitor(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        if self._count == 0:
            monitor = model.monitor
            # TODO: make Monitor support input-less channels so this hack
            # isn't necessary
            hack = monitor.channels.values()[0]
            for var in self._vars:
                monitor.add_channel(name=var.name, val=var,
                                    ipt=hack.graph_input, dataset=hack.dataset)


        if self._count in self._epoch_to_updates:
            for update in self._epoch_to_updates[self._count]:
                var, val = update
                var.set_value(np.cast[var.dtype](val))
        self._count += 1

class ChannelSmoother(TrainExtension):
    """
    Makes a smoothed version of a monitoring channel by averaging together
    the k most recent values of that channel.
    This is a little bit dangerous because if other TrainExtensions depend
    on the channel being up to date they must appear after this one in the
    extensions list. A better long term solution would be to make the Monitor
    support this kind of channel directly instead of hacking it in.
    Note that the Monitor will print this channel as having a value of -1, and
    then the extension will print the right value.

    Parameters
    ----------
    channel_to_smooth : WRITEME
    channel_to_publish : WRITEME
    k : WRITEME
    """

    def __init__(self, channel_to_smooth, channel_to_publish, k=5):
        self.__dict__.update(locals())
        del self.self

    def setup(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        monitor = model.monitor
        channels = monitor.channels
        channel_to_smooth = channels[self.channel_to_smooth]
        ipt = channel_to_smooth.graph_input
        dataset = channel_to_smooth.dataset

        monitor.add_channel(name=self.channel_to_publish,
                ipt=ipt,
                val=-1.,
                dataset=dataset)

        self.in_ch = channel_to_smooth
        self.out_ch = channels[self.channel_to_publish]

    def on_monitor(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        val_record = self.in_ch.val_record

        start = max(0, len(val_record) - self.k + 1)
        values = val_record[start:]
        mean = sum(values) / float(len(values))

        self.out_ch.val_record[-1] = mean
        print '\t' + self.channel_to_publish + ': ' + str(mean)
