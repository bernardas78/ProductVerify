from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
import math
from keras import backend as K
import os
import pandas as pd
from matplotlib import pyplot as plt

from centerLoss_MNIST import run
from Globals.globalvars import Glb

#data_dir = Glb.images_balanced_folder
cnt_classes = 10

prelast_size = 2

dists_filename = os.path.join ( Glb.results_folder, "dists_mnist_{}.csv".format(prelast_size) )
df = pd.DataFrame (columns = ["correct", "dist"] )
df.to_csv(dists_filename, mode="w", header=True, index=False)

def dists_to_center_sqsum(centers, prelast_activations):
    #   centers: (194 x 128)
    #   prelast_activations: (m x 128)
    # Returns:
    #   dists_sq from each centers (m x 194)

    m = prelast_activations.shape[0]
    cnt_classes_this = centers.shape[0]
    dists = np.zeros( (m,cnt_classes_this), dtype='float16')

    for i in range(m):
        datapoint = prelast_activations[i,:]
        delta_datapoint = centers - datapoint
        dists [i,:] = np.sum(delta_datapoint**2, axis=1)
    return dists

def visualize_cl():
    df = pd.read_csv(dists_filename, header=0)
    true_lbl = "True: {:.3f}+/-{:.3f}".format ( np.mean(df[df.correct==1].dist), np.std(df[df.correct==1].dist) )
    false_lbl = "False: {:.3f}+/-{:.3f}".format ( np.mean(df[df.correct==0].dist), np.std(df[df.correct==0].dist) )
    plt.hist( np.repeat(df[df.correct==1].dist, 9), bins=50, color="g", alpha=0.3, label=true_lbl) #repeat 9 times so that #true = #false
    plt.hist( df[df.correct==0].dist, bins=50, color="r", alpha=0.3, label=false_lbl)
    plt.legend()
    plt.savefig( os.path.join ( Glb.results_folder, 'dists_mnist_{}.png'.format(prelast_size)))
    plt.close()
    #print (true_lbl)
    #print (false_lbl)

# Load model
model_cl = run(0.5)
#model_cl_filename = os.path.join(Glb.results_folder, "Models", "model_centerloss_{}.h5".format (model_cl_date) )
#print ("Loading {}".format(model_cl_filename))
#model_cl = load_model( model_cl_filename , custom_objects={'CenterLossLayer': centerLossLayer.CenterLossLayer, 'center_loss': centerLossLayer.center_loss} )
#print ("Loaded")

# centerloss layer
cl_layer = model_cl.get_layer('centerlosslayer')
print ("cl_layer: {}".format(cl_layer))

#prelast layer function
#prelast_layer = model_cl.get_layer('activation_8')
for layer_before_centerloss in cl_layer._inbound_nodes[0].inbound_layers:
    if (type(layer_before_centerloss).__name__ != 'InputLayer'):
        prelast_layer = layer_before_centerloss
        print ("prelast_layer: {}".format(prelast_layer))
func_prelast = K.function( [model_cl.input], [prelast_layer.output] )

# get center values
centers = cl_layer.get_weights()[0]
assert (centers.shape==(cnt_classes,prelast_size))

# load data
#data_dir_train10 = os.path.join(data_dir, "Train10")
#train_iterator = MyIterator(data_dir_train10)
#data_yielder = train_iterator.get_iterator_xy_ydummy()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("type(x_train):{}".format(type(x_train)))

x_train = x_train.reshape((-1, 28, 28, 1))
y_train_onehot = to_categorical(y_train, 10)
batch_size=64

# compute distance from center and log to file
for batch_id in range(int(x_train.shape[0]/batch_size)):
    # get data
    x_y = (x_train[batch_id*batch_size:(batch_id+1)*batch_size,:,:,:], y_train_onehot[batch_id*batch_size:(batch_id+1)*batch_size,:])
    y_dummy = (y_train_onehot[batch_id*batch_size:(batch_id+1)*batch_size,:], np.zeros((batch_size, 1)) )
    assert ( x_y[0].shape[1:4] == (28,28,1) )
    assert ( x_y[1].shape[1:2] == (cnt_classes,) )
    # calc dist to center
    prelast_output = func_prelast(x_y)[0]
    assert (prelast_output.shape[1:2] == (prelast_size,))
    dists_to_centers_batch = dists_to_center_sqsum (centers = centers, prelast_activations=prelast_output)
    assert (dists_to_centers_batch.shape[1:2] == (cnt_classes,))

    # output 2 1D arrays to file: correct;dist
    y_ravel = np.ravel(x_y[1])
    dist_ravel = np.ravel(dists_to_centers_batch)
    df = pd.DataFrame ()
    df['correct'] = y_ravel
    df['dist'] = dist_ravel
    df.to_csv(dists_filename, mode="a", header=False, index=False)

# sanity check
print ("Sanity check: L2 norms calculated using [prelast_output-centers] vs. centerloss")
_, cl = model_cl.predict(x_y)
y = x_y[1]
for i in range( y.shape[0] ):
    Y = np.argmax(y[i])
    dist_center = prelast_output[i] - centers[Y]
    dist_center_sqsum = np.sum ( np.dot(dist_center,dist_center) )
    print ( "dist_center_sqsum:{}; cl:{}".format(dist_center_sqsum, cl[i][0]))
    #assert math.isclose(dist_center_sqsum, cl[i][0], rel_tol=1e-4)

visualize_cl()