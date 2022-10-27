from tensorflow.keras.layers import Input
from tensorflow import keras
from centerLossLayer import CenterLossLayer

def make_model_cl (model_clsf):

    Softmax_size = model_clsf.output_shape[1]
    #print ("Softmax_size: {}".format(Softmax_size))

    # second input - labels
    labels_input = Input((Softmax_size,))

    # pre-last layer's output
    x = model_clsf.layers[-2].output

    PreLastDense_size = x.shape[1]
    #print ("PreLastDense_size: {}".format(PreLastDense_size))

    output_centerLoss = CenterLossLayer(Softmax_size=Softmax_size, PreLastDense_size=PreLastDense_size, alpha=0.5, name='centerlosslayer')([x,labels_input])

    model_cl = keras.Model(inputs=[model_clsf.inputs,labels_input], outputs=[model_clsf.output,output_centerLoss])

    return model_cl