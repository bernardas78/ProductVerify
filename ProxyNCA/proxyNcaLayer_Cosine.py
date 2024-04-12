from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class ProxyNcaLayer(Layer):

    def __init__(self, Softmax_size=194, PreLastDense_size=128, alpha=0.5, p=0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.Softmax_size = Softmax_size
        self.PreLastDense_size = PreLastDense_size
        print("ProxyNcaLayer_Cosine.init")

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

        #print ("self.centers: {}".format(self.centers))
        ##################
        # calc new centers
        ##################

        # normalize to unit length
        x[0] = x[0] / K.sqrt ( K.sum(x[0] ** 2, axis=1, keepdims=True) )
        #print ("x[0]: {}".format(x[0]))
        #print ("x[1]: {}".format(x[1]))

        # x[0] is mx2, x[1] is mx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers

        # normalize to unit length
        new_centers = new_centers / K.sqrt ( K.sum(new_centers ** 2, axis=1, keepdims=True) )
        #print ("new_centers: {}".format(new_centers))

        ##################
        # update centers
        ##################
        self.add_update(K.moving_average_update(self.centers, new_centers, 0.0))
        #self.add_update((self.centers, new_centers))#, x)
        #print ("self.centers: {}".format(self.centers))

        #print ("centers updated. shape: {}".format(new_centers.shape))

        ##################
        # calc loss
        ##################

        s = K.dot ( x[0], K.transpose(self.centers) ) # mx10
        s = s - 1 # range [-2:0]
        s_pos = s * x[1]
        s_neg = s * (1.-x[1])

        #print ("s:{}".format(s))
        #print ("s_pos:{}".format(s_pos))
        #print ("neg:{}".format(s_neg))

        # LSE - log-sum-exp of negative (-1 compenssates for exp(0) of the positive center)
        self.result = - K.sum(s_pos, axis=1, keepdims=True) +  K.log ( K.sum ( K.exp(s_neg), axis=1, keepdims=True )-1.0 )
        #print ("Shape: {}".format(K.eval(self.result.shape)))

        return self.result # mx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def proxynca_loss(y_true, y_pred):
    #print("SHAPE: {}".format(K.eval(y_pred.shape)))
    return K.sum(y_pred, axis=0)
