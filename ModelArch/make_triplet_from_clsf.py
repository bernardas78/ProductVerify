
from tensorflow.keras.layers import Input, Lambda, Dense, Activation, Subtract, Layer
from tensorflow import keras
from Layers.triplet_distanceLayer import DistanceLayer


def make_model_triplet (model_clsf, cnt_trainable, distName):
    # from https://keras.io/examples/vision/siamese_network/

    anchor_input  = Input(name="anchor",  shape=model_clsf.layers[0].input.shape[1:] )
    positive_input  = Input(name="positive",  shape=model_clsf.layers[0].input.shape[1:] )
    negative_input  = Input(name="negative",  shape=model_clsf.layers[0].input.shape[1:] )

    clsf_input = model_clsf.layers[0].input

    # pre-last layer's output
    prelast_output = model_clsf.layers[-2].output
    #prelast_output = Dense(512, name='DenseBeforeSubtract')(prelast_output)

    embedding_network = keras.Model(clsf_input, prelast_output, name="embeddingNet")

    #cnt_trainable = 8 #dense+batchNorm+dropout+activation - 2 blocks
    #cnt_trainable = 4 #dense+batchNorm+dropout+activation - last block

    for i,layer in enumerate (embedding_network.layers):
        #print ("i={}, len={}".format(i,len(embedding_network.layers)))
        if (i+cnt_trainable) < len(embedding_network.layers):
            layer.trainable = False
            #print ("untrainable layer {}".format(layer.name))


    distances = DistanceLayer(distName)()(
        embedding_network(anchor_input),
        embedding_network(positive_input),
        embedding_network(negative_input)
    )

    model_triplet = keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)

    #for layer in model_triplet.layers:
        #print ("layer {} trainable={}".format(layer.name,layer.trainable))
        #layer.trainable = False
        #print ("layer {} trainable={}".format(layer.name,layer.trainable))

    return model_triplet
