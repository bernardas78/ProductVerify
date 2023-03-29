import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Activation, Subtract
from tensorflow import keras

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def make_model_siam (model_clsf):
    # from https://keras.io/examples/vision/siamese_contrastive/

    # pre-last layer's output
    prelast_output = model_clsf.layers[-2].output
    #prelast_output = model_clsf.layers[-6].output


    prelast_output = Dense(512, activation='relu', name='DenseBeforeSubtract')(prelast_output)
    #prelast_output = Dense(256, activation='relu', name='DenseBeforeSubtract2')(prelast_output)
    #prelast_output = model_clsf.layers[-6].output #remove dense
    #prelast_output = Dense(512, name='Dense_post_originally_prelast')(prelast_output)  #add dense

    clsf_input = model_clsf.layers[0].input

    embedding_network = keras.Model(clsf_input, prelast_output)

    #for layer in embedding_network.layers:
    #    layer.trainable = False

    input_1 = Input( model_clsf.layers[0].input.shape[1:] )
    input_2 = Input( model_clsf.layers[0].input.shape[1:] )

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    #merge_layer = Subtract()([tower_1, tower_2])
    merge_layer = Lambda(euclidean_distance, name="lambda_layer")([tower_1, tower_2])

    merge_layer = keras.layers.BatchNormalization(name="batchnorm_layer")(merge_layer)
    #output_layer = Dense(1, activation="sigmoid", name="sigmoid_layer")(normal_layer)

    #merge_layer = Dense(96, activation="relu", name="relu_layer")(merge_layer)
    #merge_layer = Dense(64, activation="relu", name="relu_layer1")(merge_layer)

    #output_layer = Dense(1, activation="sigmoid", name="sigmoid_layer")(merge_layer)
    #output_layer = Activation('sigmoid')(merge_layer)
    output_layer = merge_layer

    model_siam = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    return model_siam
