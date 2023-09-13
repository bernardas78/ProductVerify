# branch of make_cl_from_clsf.py
#   add extra Dense after Dense(128) so that centers are smaller

from tensorflow.keras.layers import Input, Dense
from tensorflow import keras
from CenterLoss.centerLossLayer import CenterLossLayer

def make_model_cl (model_clsf, Softmax_size, dense_size, distName, p_minkowski, inclInterCenter, lambda2, pre_cl_layer_ind=0):

    #Softmax_size = model_clsf.output_shape[1]
    #print ("Softmax_size: {}".format(Softmax_size))

    # second input - labels
    labels_input = Input((Softmax_size,), name='input_labels')

    # pre-last layer's output
    x = model_clsf.layers[-2].output

    # add extra dense
    x = Dense(dense_size, name='Dense_post_originally_prelast')(x)

    # add softmax
    out_softmax = Dense(Softmax_size, activation='softmax', name='DenseSoftmax')(x)

    # 0-use extra dense; -x:
    if pre_cl_layer_ind==0:
        pre_cl_layer_output = x
    else:
        pre_cl_layer_output = model_clsf.layers[pre_cl_layer_ind].output

    PreClDense_size = pre_cl_layer_output.shape[1]
    #print ("PreLastDense_size: {}".format(PreLastDense_size))

    output_centerLoss = CenterLossLayer\
        (distName,inclInterCenter)\
        (Softmax_size=Softmax_size, PreLastDense_size=PreClDense_size, alpha=0.5, name='centerlosslayer', p=p_minkowski, lambda2=lambda2)\
        ([pre_cl_layer_output,labels_input])

    model_cl = keras.Model(inputs=[model_clsf.inputs,labels_input], outputs=[out_softmax,output_centerLoss])

    return model_cl