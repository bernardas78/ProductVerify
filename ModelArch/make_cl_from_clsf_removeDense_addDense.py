# branch of make_cl_from_clsf.py
#   add extra Dense after Dense(128) so that centers are smaller

from tensorflow.keras.layers import Input, Dense
from tensorflow import keras
from centerLossLayer import CenterLossLayer

def make_model_cl (model_clsf, dense_size=64):

    Softmax_size = model_clsf.output_shape[1]
    #print ("Softmax_size: {}".format(Softmax_size))

    # second input - labels
    labels_input = Input((Softmax_size,), name='input_labels')

    # pre-pre-last layer's output
    x = model_clsf.layers[-6].output

    # add extra dense
    x = Dense(dense_size, name='Dense_post_originally_prelast')(x)

    PreLastDense_size = x.shape[1]
    #print ("PreLastDense_size: {}".format(PreLastDense_size))

    # add softmax
    out_softmax = Dense(Softmax_size, activation='softmax', name='DenseSoftmax')(x)

    output_centerLoss = CenterLossLayer(Softmax_size=Softmax_size, PreLastDense_size=PreLastDense_size, alpha=0.5, name='centerlosslayer')([x,labels_input])

    model_cl = keras.Model(inputs=[model_clsf.inputs,labels_input], outputs=[out_softmax,output_centerLoss])

    return model_cl