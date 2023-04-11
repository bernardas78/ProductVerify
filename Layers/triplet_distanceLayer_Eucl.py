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
        print ("Distance type: Eucl")

    def call(self, anchor, positive, negative):
        ap_distance = tf.sqrt ( tf.reduce_sum(tf.square(anchor - positive), -1) )
        an_distance = tf.sqrt ( tf.reduce_sum(tf.square(anchor - negative), -1) )
        #return (ap_distance, an_distance)
        return ap_distance - an_distance

def dist_func(emb_a, emb_b):
    return np.sqrt(np.sum((emb_a - emb_b) ** 2, axis=1))