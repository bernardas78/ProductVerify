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
        print ("Distance type: Manh")

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.abs(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.abs(anchor - negative), -1)
        #return (ap_distance, an_distance)
        return ap_distance - an_distance