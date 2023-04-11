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
        print ("Distance type: Cosine")


    def call(self, anchor, positive, negative):
        anchor_norm = tf.nn.l2_normalize(anchor,-1)
        positive_norm = tf.nn.l2_normalize(positive,-1)
        negative_norm = tf.nn.l2_normalize(negative,-1)

        ap_distance = - tf.reduce_sum ( tf.multiply(anchor_norm,positive_norm), -1)
        an_distance = - tf.reduce_sum ( tf.multiply(anchor_norm,negative_norm), -1)

        #return (ap_distance, an_distance)
        return ap_distance - an_distance

def dist_func(emb_a, emb_b):
    emb_a_norm = emb_a / np.maximum ( np.linalg.norm(emb_a, axis=-1, keepdims=True), 1e-7)
    emb_b_norm = emb_b / np.maximum ( np.linalg.norm(emb_b, axis=-1, keepdims=True), 1e-7)
    return 1. - np.sum( np.multiply(emb_a_norm, emb_b_norm), axis=1)