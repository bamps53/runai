import keras.layers
import tensorflow as tf

import runai

from . import coordinator

class Parallelised(keras.layers.Layer):
    """ A helper class for MP-supported Keras layers
    """

    def add_weights(self, name, shape, **kwargs):
        """ Declare parallelised weights

        # Arguments:
            name: The base name of the weights
            shape: The (split) shape of each weight
        
        # Returns:
            A list of tensors with the tensor names of the weights
        """

        runai.log.debug('Declaring %d weights (%s) of shape %s for \'%s\' layer "%s"' % (runai.mp.splits, name, shape, self.__class__.__name__, getattr(self, 'name', 'N/A')))

        def add_weight(gpu):
            with tf.device('/device:GPU:%d' % gpu):
                return self.add_weight(
                    name='%s_%d' % (name, gpu),
                    shape=shape,
                    **kwargs)
        
        return [add_weight(gpu) for gpu in range(runai.mp.splits)]

    def calculate_cs(self, cin, cout):
        """ Calculate the 'C' dimensions

        Using the total (i.e. original) cin and cout dimension sizes,
        we calculate the split sizes of those dimensions. In addition,
        we provide an extra dimension size (c) which is the split size
        of the output dimension.

        # Arguments:
            cin: the total cin size
            cout: the total cout size
        
        # Returns:
            A tuple (cin, cout, c) of the per-GPU sizes
        """

        # TODO(levosos): support uneven division
        
        c = cout // runai.mp.splits # output dimension is always split

        if runai.mp.method == runai.mp.Method.Cin:
            cin = cin // runai.mp.splits
        elif runai.mp.method == runai.mp.Method.Cout:
            cout = c
        else:
            raise ValueError('Unrecognized MP method: %s' % runai.mp.method)
        
        return (cin, cout, c)

    def inputs(self, input, channel_axis):
        """ Get the respective split (per-GPU) inputs for an input tensor

        Returns the split inputs of a merged tensor in the correct way.
        This varies between the MP methods.
        """

        assert isinstance(input, tf.Tensor) # and not a list/tuple of tensors

        if runai.mp.method == runai.mp.Method.Cin:
            if coordinator.registered(input):
                runai.log.info('Using parallelised input for \'%s\' layer "%s"', self.__class__.__name__, getattr(self, 'name', 'N/A'))
                return coordinator.resolve(input)
            else:
                runai.log.warning('Splitting non-parallelised input (%s) for \'%s\' layer "%s"', input.name, self.__class__.__name__, getattr(self, 'name', 'N/A'))
                return tf.split(input, runai.mp.splits, axis=channel_axis)
        elif runai.mp.method == runai.mp.Method.Cout:
            return [input] * runai.mp.splits
        else:
            raise ValueError('Unrecognized MP method: %s' % runai.mp.method)

    def merge(self, outputs, channel_axis):
        """ Merge and register the split outputs of the layer

        Merges a parallelised layer's output tensors and register
        then in 'coordinator' for later use.
        The merge is done be concatenating the tensors on the specified
        channel axis.

        # Returns:
            A tensor representing the merged outputs of the layer
        """

        merged = keras.layers.Concatenate(axis=channel_axis)(outputs)
        coordinator.register(merged, outputs)
        return merged

    def parallelise(self, l, *iterables):
        """ Evaluate a lambda on each GPU with device placement
        """
        
        def wrap(gpu, *args):
            with tf.device('/device:GPU:%d' % gpu):
                return l(*args)

        return [wrap(gpu, *args) for gpu, args in enumerate(zip(*iterables))]

    def reduce_split(self, outputs, channel_axis):
        """ Reduce-split some tensors on a specified axis
        """

        # TODO(levosos): implement better reduce-split

        added = keras.layers.Add()(outputs)
        return tf.split(added, runai.mp.splits, axis=channel_axis)
