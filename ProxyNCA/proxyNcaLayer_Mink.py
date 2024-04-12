from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

class ProxyNcaLayer(Layer):

    def __init__(self, Softmax_size=194, PreLastDense_size=128, alpha=0.5, p=1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.Softmax_size = Softmax_size
        self.PreLastDense_size = PreLastDense_size
        self.p = p
        print("ProxyNcaLayer_Mink.init (p={})".format(p))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha,
            'Softmax_size': self.Softmax_size,
            'PreLastDense_size': self.PreLastDense_size,
            'p': self.p
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

        # x[0] is mx2, x[1] is mx10 onehot, self.centers is 10x2
        #print ("x[0]:{}".format(x[0]))
        #print ("self.centers:{}".format(self.centers))

        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        #print("delta_centers: {}".format(delta_centers))
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers

        # normalize to unit length
        #new_centers = new_centers / K.sqrt ( K.sum(new_centers ** 2, axis=1, keepdims=True) )
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

        m = x[0].shape[0]
        #print ("m={}".format(m))

        # PRINT x[0]
        #print ("x[0]:{}".format(x[0]))
        #print ("tf.permute_dimensions...:{}".format(K.permute_dimensions ( tf.broadcast_to(x[0],(self.Softmax_size,m,self.PreLastDense_size)), (1,0,2))))
        #print ("tf.permute_....shape:{}".format(K.permute_dimensions ( tf.broadcast_to(x[0],(self.Softmax_size,m,self.PreLastDense_size)), (1,0,2)).shape))

        # PRINT self.centers
        #print ("self.centers: {}".format(self.centers))
        #print("tf.broadcast...{}".format(tf.broadcast_to(self.centers,(m,self.Softmax_size,self.PreLastDense_size))))
        #print("tf.broadcast..shape:{}".format(tf.broadcast_to(self.centers,(m,self.Softmax_size,self.PreLastDense_size)).shape))

        #x0_broad = tf.broadcast_to(x[0], (self.Softmax_size, m, self.PreLastDense_size))
        x0_broad = K.repeat_elements ( K.expand_dims(x[0], axis=1), rep=self.Softmax_size, axis=1 )
        #print ("x0_broad.shape: {}".format(x0_broad.shape))
        #print ("x0_broad: {}".format(x0_broad))

        #centers_broad = tf.broadcast_to(self.centers,(m,self.Softmax_size,self.PreLastDense_size))
        #centers_broad = K.repeat_elements ( K.expand_dims(self.centers, axis=0), rep=m, axis=0 )
        centers_broad = K.expand_dims(self.centers, axis=0)
        #print ("centers_broad.shape: {}".format(centers_broad.shape))
        #print ("centers_broad: {}".format(centers_broad))


        #D = K.permute_dimensions ( x0_broad, (1,0,2)) - centers_broad
        D = x0_broad - centers_broad
        # D.shape = (m,n,PreLastDense_size)
        assert (D.shape==(m,self.Softmax_size, self.PreLastDense_size) )
        #print ("D:{}".format(D))

        # d.shape = (m,n)
        d = K.pow( K.sum ( K.pow( K.abs (D),self.p), axis=2), 1./self.p )
        assert (d.shape == (m,self.Softmax_size) )

        s = -d
        s_pos = s * x[1]
        s_neg = s * (1.-x[1])

        #print ("s:{}".format(s))
        #print ("s_pos:{}".format(s_pos))
        #print ("neg:{}".format(s_neg))

        # LSE - log-sum-exp of negative
        self.result = - K.sum(s_pos, axis=1, keepdims=True) +  K.log ( K.sum ( K.exp(s_neg), axis=1, keepdims=True ) )
        #print ("Shape: {}".format(K.eval(self.result.shape)))

        return self.result # mx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def proxynca_loss(y_true, y_pred):
    #print("SHAPE: {}".format(K.eval(y_pred.shape)))
    return K.sum(y_pred, axis=0)
