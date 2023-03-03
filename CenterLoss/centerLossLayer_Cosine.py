from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class CenterLossLayer(Layer):

    def __init__(self, Softmax_size=194, PreLastDense_size=128, alpha=0.5, p=0, lambda2=1., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.Softmax_size = Softmax_size
        self.PreLastDense_size = PreLastDense_size
        print("centerLossLayer_Cosine.init")

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha,
            'Softmax_size': self.Softmax_size,
            'PreLastDense_size': self.PreLastDense_size,
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

        # x[0] - preCL activations; m * precl_size
        # x[1] - labels, onehot;    m * n
        # self.centers;             n * precl_size

        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers))#, x)
        #print ("centers updated. shape: {}".format(new_centers.shape))

        # this_batch_centers - centers of this batch' samples;   m * precl_size
        this_batch_centers = K.dot(x[1], self.centers)
        #       cosine distance [0;2]:
        #               dot product (unnormalized cosine silimilarity)
        #               divide by: len(x[0]), len(this_batch_centers)
        #               1-cosine silimilarity
        self.result = K.sum ( x[0] * this_batch_centers, axis=1, keepdims=True )
        self.result = self.result / K.sqrt( K.sum(x[0] ** 2, axis=1, keepdims=True) )
        self.result = self.result / K.sqrt(K.sum(this_batch_centers ** 2, axis=1, keepdims=True))
        self.result = 1. - self.result

        return self.result # mx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def center_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)
