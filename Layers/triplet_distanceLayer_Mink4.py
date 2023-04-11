import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class DistanceLayer(Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print ("Distance type: Mink, p=4")

    def call(self, anchor, positive, negative):
        ap_distance = tf.pow ( tf.reduce_sum(tf.pow( tf.abs (anchor - positive), 4), -1), 1/4)
        an_distance = tf.pow ( tf.reduce_sum(tf.pow( tf.abs (anchor - negative), 4), -1), 1/4)
        #return (ap_distance, an_distance)
        return ap_distance - an_distance

def dist_func(emb_a, emb_b):
    return np.power(np.sum( np.power(emb_a - emb_b, 4), axis=1), 1/4)
