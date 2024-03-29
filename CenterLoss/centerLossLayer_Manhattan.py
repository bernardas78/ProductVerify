from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class CenterLossLayer(Layer):

    def __init__(self, Softmax_size=194, PreLastDense_size=128, alpha=0.5, p=0, lambda2=1.,  **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.Softmax_size = Softmax_size
        self.PreLastDense_size = PreLastDense_size
        print("centerLossLayer_Manhattan.init")

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

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers))#, x)
        #print ("centers updated. shape: {}".format(new_centers.shape))

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum( K.abs( self.result ) , axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def center_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)
