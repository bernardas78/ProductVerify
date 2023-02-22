#from tensorflow.python.keras.layers import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class CenterLossLayer(Layer):

    def __init__(self, Softmax_size=194, PreLastDense_size=128, alpha=0.5, p=0, lambda2=1., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.Softmax_size = Softmax_size
        self.PreLastDense_size = PreLastDense_size
        self.lambda2 = lambda2
        print("centerLossLayer_Eucl_InterC.old.init.lambda2:{}".format(lambda2))

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

        # x[0] is mx2, x[1] is mx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers))#, x)
        #print ("centers updated. shape: {}".format(new_centers.shape))

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sqrt ( K.sum(self.result ** 2, axis=1, keepdims=True) )#/ K.dot(x[1], center_counts)

        interc_loss = interc_dist_mean(self.centers)

        return self.result + self.lambda2 * interc_loss # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def center_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)

def interc_dist_mean (c):
    # c is [Softmax_size * PreLastDense_size]
    c_shape = K.int_shape(c)

    # dot(v1,v2) - vector similarity (~cosine sim, not normalized)
    c_dot = K.dot (c, K.transpose(c) ) # c_dot is [Softmax_size * Softmax_size]

    # length of centers
    c_len = K.sqrt ( K.sum ( c * c, axis=1, keepdims=True ) ) # c_len   is [Softmax_size * 1]
    #                                                           c_len.T is [1 * Softmax_size]
    #                                                    ( c_len * K.transpose (c_len)) is [Softmax_size * Softmax_size]

    # 1 + cosine distance; range [0;2]
    cosine_dist = 1 - c_dot / ( c_len * K.transpose (c_len) ) # cosine_dist is [Softmax_size * Softmax_size]

    # cosine_dist has zero diagonal (self-dots); total n*(n-1) non-zero members
    #   range [0;2]
    mean_interc_loss = 2 - K.sum(cosine_dist) / ( c_shape[0] * ( c_shape[0]-1 ) ) # mean_interc_loss is scalar

    return mean_interc_loss

