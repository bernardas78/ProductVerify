#from tensorflow.python.keras.layers import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

class CenterLossLayer(Layer):

    def __init__(self, Softmax_size=194, PreLastDense_size=128, alpha=0.5, p=0, lambda2=1., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.Softmax_size = Softmax_size
        self.PreLastDense_size = PreLastDense_size
        self.lambda2 = lambda2
        print("centerLossLayer_Eucl_InterC.init.lambda2:{}".format(lambda2))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha,
            'Softmax_size': self.Softmax_size,
            'PreLastDense_size': self.PreLastDense_size,
            'lambda2': self.lambda2,
        })
        return config

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.Softmax_size, self.PreLastDense_size),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):
        # x[0] - pre-last layer activations; m * PreLastDense_size
        # x[1] - labels;                     m * Softmax_size onehot
        # self.centers                       Softmax_size * PreLastDense_size
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # Softmax_size * PreLastDense_size
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # Softmax_size * 1
        delta_centers /= center_counts

        ic_delta_centers = interc_delta(self.centers, self.lambda2)

        new_centers = self.centers - self.alpha * delta_centers + ic_delta_centers
        self.add_update((self.centers, new_centers))#, x)
        #print ("centers updated. shape: {}".format(new_centers.shape))

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers) + K.sum(ic_delta_centers)
        self.result = K.sqrt ( K.sum(self.result ** 2, axis=1, keepdims=True) )#/ K.dot(x[1], center_counts)

        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def center_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)

def interc_delta (C, lambda2):
    # c is [Softmax_size * PreLastDense_size]
    c_shape = K.int_shape(C)

    # dot(v1,v2) - vector similarity (~cosine sim, not normalized)
    c_dot = K.dot (C, K.transpose(C) ) # c_dot is [Softmax_size * Softmax_size]

    # length of centers
    c_len = K.sqrt ( K.sum ( C * C, axis=1, keepdims=True ) ) # c_len   is [Softmax_size * 1]
    #                                                           c_len.T is [1 * Softmax_size]
    #                                                    ( c_len * K.transpose (c_len)) is [Softmax_size * Softmax_size]

    # cosine_sim is [Softmax_size * Softmax_size]; range [0,2]
    cosine_sim = 1 + c_dot / ( c_len * K.transpose (c_len) )

    # sim = diagonal-removed-cosine_sim
    sim = ( 1 - tf.eye( c_shape[0]) ) * cosine_sim
    #print ("sim:{}".format(sim))

    # intercenter_delta.shape [Softmax_size * PreLastDense_size]
    intercenter_delta = K.sum(sim, axis=1, keepdims=True) * C - K.dot ( sim, C )
    #print ("line 81:{}".format(intercenter_delta))

    # experimentally after such normalization:, max(intercenter_delta)~max(C)
    intercenter_delta = intercenter_delta /  c_shape[0]
    #print ("line 84:{}".format(intercenter_delta))

    # to keep center lengths the same
    c_newdir = C + intercenter_delta * lambda2
    #print ("line 88:{}".format(c_newdir))
    c_newdir_len = K.sqrt ( K.sum ( c_newdir * c_newdir, axis=1, keepdims=True ) )  # [Softmax_size * 1]
    #print ("line 90:{}".format(c_newdir_len))
    intercenter_delta = c_newdir * ( c_len / c_newdir_len ) - C

    return intercenter_delta

