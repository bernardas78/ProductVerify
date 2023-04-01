import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Activation, Subtract, Layer
from tensorflow import keras

class DistanceLayer(Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        #return (ap_distance, an_distance)
        return ap_distance - an_distance

def make_model_triplet (model_clsf):
    # from https://keras.io/examples/vision/siamese_network/

    anchor_input  = Input(name="anchor",  shape=model_clsf.layers[0].input.shape[1:] )
    positive_input  = Input(name="positive",  shape=model_clsf.layers[0].input.shape[1:] )
    negative_input  = Input(name="negative",  shape=model_clsf.layers[0].input.shape[1:] )

    clsf_input = model_clsf.layers[0].input

    # pre-last layer's output
    prelast_output = model_clsf.layers[-2].output
    #prelast_output = Dense(512, name='DenseBeforeSubtract')(prelast_output)

    embedding_network = keras.Model(clsf_input, prelast_output)

    distances = DistanceLayer()(
        embedding_network(anchor_input),
        embedding_network(positive_input),
        embedding_network(negative_input)
    )

    model_triplet = keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)

    return model_triplet
