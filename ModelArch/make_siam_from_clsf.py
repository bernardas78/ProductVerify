import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Activation, Subtract
from tensorflow import keras

def euclidean_distance(vects):
    print ("euclidean_distance")
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def manhattan_distance(vects):
    print ("manhattan_distance")
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.abs(x - y), axis=1, keepdims=True)
    return tf.math.maximum(sum_square, tf.keras.backend.epsilon())

def mink3_distance(vects):
    print ("mink3_distance")
    p=3.
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.pow(tf.math.abs(x - y), p), axis=1, keepdims=True)
    return tf.math.pow(tf.math.maximum(sum_square, tf.keras.backend.epsilon()), 1/p)

def mink4_distance(vects):
    print ("mink4_distance")
    p=4.
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.pow(tf.math.abs(x - y), p), axis=1, keepdims=True)
    return tf.math.pow(tf.math.maximum(sum_square, tf.keras.backend.epsilon()), 1/p)

def cosine_distance(vects):
    print ("cosine_distance")
    #print("type(vects): {}".format(type(vects)))
    #print ("len(vects): {}".format(len(vects)))
    #print ("vects[0].shape: {}".format(vects[0].shape))
    #print ("vects[1].shape: {}".format(vects[1].shape))
    #x,y = vects
    x = vects[0]
    y = vects[1]
    #print ("post x,y = vects")
    x_norm = tf.nn.l2_normalize(x, -1)
    y_norm = tf.nn.l2_normalize(y, -1)
    return - tf.reduce_sum(tf.multiply(x_norm, y_norm), -1, keepdims=True)

def make_model_siam (model_clsf, cnt_trainable, distName):
    # from https://keras.io/examples/vision/siamese_contrastive/

    # pre-last layer's output
    prelast_output = model_clsf.layers[-2].output
    #prelast_output = model_clsf.layers[-6].output


    #prelast_output = Dense(512, activation='relu', name='DenseBeforeSubtract')(prelast_output)
    prelast_output = Dense(512, name='DenseBeforeSubtract')(prelast_output)


    #prelast_output = Dense(256, activation='relu', name='DenseBeforeSubtract2')(prelast_output)
    #prelast_output = model_clsf.layers[-6].output #remove dense
    #prelast_output = Dense(512, name='Dense_post_originally_prelast')(prelast_output)  #add dense

    clsf_input = model_clsf.layers[0].input

    embedding_network = keras.Model(clsf_input, prelast_output)

    for i,layer in enumerate (embedding_network.layers):
        if (i+cnt_trainable) < len(embedding_network.layers):
            layer.trainable = False


    #for layer in embedding_network.layers:
    #    layer.trainable = False

    input_1 = Input( model_clsf.layers[0].input.shape[1:] )
    input_2 = Input( model_clsf.layers[0].input.shape[1:] )

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    #merge_layer = Subtract()([tower_1, tower_2])

    if distName=="Eucl":
        f_distance = euclidean_distance
    elif distName=="Manh":
        f_distance = manhattan_distance
    elif distName == "Mink3":
        f_distance = mink3_distance
    elif distName=="Mink4":
        f_distance = mink4_distance
    elif distName=="Cosine":
        f_distance = cosine_distance

    #merge_layer = Lambda(f_distance, name="lambda_layer")([tower_1, tower_2])
    merge_layer = Lambda(cosine_distance, name="lambda_layer")([tower_1, tower_2])

    merge_layer = keras.layers.BatchNormalization(name="batchnorm_layer")(merge_layer)
    #output_layer = Dense(1, activation="sigmoid", name="sigmoid_layer")(normal_layer)

    #merge_layer = Dense(96, activation="relu", name="relu_layer")(merge_layer)
    #merge_layer = Dense(64, activation="relu", name="relu_layer1")(merge_layer)

    output_layer = Dense(1, activation="sigmoid", name="sigmoid_layer")(merge_layer)
    #output_layer = Activation('sigmoid')(merge_layer)
    #output_layer = merge_layer

    model_siam = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    return model_siam
