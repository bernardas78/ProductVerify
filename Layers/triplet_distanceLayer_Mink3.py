import tensorflow as tf
from tensorflow.keras.layers import Layer

class DistanceLayer(Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print ("Distance type: Mink, p=3")

    def call(self, anchor, positive, negative):
        ap_distance = tf.pow ( tf.reduce_sum(tf.pow( tf.abs (anchor - positive), 3), -1), 1/3)
        an_distance = tf.pow ( tf.reduce_sum(tf.pow( tf.abs (anchor - negative), 3), -1), 1/3)
        #return (ap_distance, an_distance)
        return ap_distance - an_distance