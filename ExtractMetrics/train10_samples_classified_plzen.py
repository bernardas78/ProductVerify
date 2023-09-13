import numpy as np
import math
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import os
import pandas as pd
import sys
from trained_model_names import model_names, experSuffix_names
from PIL import Image
from matplotlib import pyplot as plt

sys.path.insert(0,'..')

from CenterLoss.centerLossLayer import CenterLossLayer, center_loss
from Globals.globalvars import Glb, MyTfrecordIterator


data_dir = Glb.images_balanced_folder
cnt_classes = 194

#set_name = "Val"
set_name = "Train10"

# from A:\IsKnown_Results\Dists\dists_512_Eucl.csv, needed to scale same as in roc calculation
dist_min = 0.531
dist_max = 1.075

# from make_auc_byDistTypeAndNeuronCount_CenterLoss.py
eer_threshold = 0.199

selected_class_inds = [2,7,45]
selected_class_names = ["Cucumber","Banana","Apple"]

experName = ""
preClIndex = -1234
prelast_size = -5423
distName = "Eucl"
p_minkowski = -1
inclInterCenter = False
lambda2 = -1234234.34543
lambda1 = "n/a"

prelast_size = 512
distName = "Eucl"
p_minkowski = -1
inclInterCenter = False
lambda2=-1

mink_suffix = "_{}".format(p_minkowski) if distName == "Minkowski" else ""
interc_suffix = "_{:.3f}".format(lambda2) if inclInterCenter else ""

tfrecord_dir = os.path.join(Glb.images_folder, "PV_TFRecord")
tfrecord_filepath = os.path.join(tfrecord_dir, "{}.tfrecords".format(set_name))

dists_filename = os.path.join ( "S:", "VerifResult", "dists.csv" )
experSuffix = experSuffix_names(distName,prelast_size,p_minkowski,inclInterCenter,lambda2,experName,preClIndex, lambda1)
#dists_filename = os.path.join ( Glb.results_folder, "Dists", "dists{}csv".format(experSuffix) )
verif_result_dir = os.path.join ( "S:", "VerifResult")
df = pd.DataFrame (columns = ["batch_sample", "correct", "dist"] )
df.to_csv(dists_filename, mode="w", header=True, index=False)

def dists_to_center_minkowskisum(centers, prelast_activations, **kwargs):
    #   centers: (194 x 128)
    #   prelast_activations: (m x 128)
    # Returns:
    #   dists_sq from each centers (m x 194)
    p = int(kwargs["p"])
    #print ("p={}".format(p))

    m = prelast_activations.shape[0]
    cnt_classes_this = centers.shape[0]
    dists = np.zeros( (m,cnt_classes_this), dtype='float16')

    for i in range(m):
        datapoint = prelast_activations[i,:]
        delta_datapoint = centers - datapoint
        dists [i,:] = np.power ( np.sum( np.power (np.abs(delta_datapoint), p), axis=1), 1/p )
    return dists

def dists_to_center_abssum(centers, prelast_activations, **kwargs):
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
        dists [i,:] = np.sum(np.abs(delta_datapoint), axis=1)
    return dists

def dists_to_center_sqsum(centers, prelast_activations, **kwargs):
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
        dists [i,:] = np.sqrt ( np.sum(delta_datapoint**2, axis=1) )
    return dists

if distName=="Eucl":
    dists_to_center_f = dists_to_center_sqsum
elif distName == "Manhattan":
    dists_to_center_f = dists_to_center_abssum
elif distName == "Minkowski":
    dists_to_center_f = dists_to_center_minkowskisum
else:
    raise Exception("Unknown distance function")

# Load model
model_cl_filename = os.path.join(Glb.results_folder, "Models", "model"+experSuffix+"h5" )
#model_cl_filename = os.path.join(Glb.results_folder, "Models", model_names(distName,prelast_size,p_minkowski,inclInterCenter,lambda2,experName,preClIndex) )
print ("Loading {}".format(model_cl_filename))
model_cl = load_model(model_cl_filename, custom_objects={'CenterLossLayer': CenterLossLayer(distName,inclInterCenter), 'center_loss': center_loss(distName)})
print ("Loaded")

# centerloss layer
cl_layer = model_cl.get_layer('centerlosslayer')
print ("cl_layer: {}".format(cl_layer))

# update minkowski coefficient
if distName=="Minkowski":
    cl_layer.p = p_minkowski
    cl_layer.get_config()
    print ("cl_layer.p: {}".format(cl_layer.p))

if inclInterCenter:
    print("Lambda2:{}".format(cl_layer.lambda2))

#prelast layer function
for layer_before_centerloss in cl_layer._inbound_nodes[0].inbound_layers:
    if (type(layer_before_centerloss).__name__ != 'InputLayer'):
        prelast_layer = layer_before_centerloss
        print ("prelast_layer: {}".format(prelast_layer))
func_prelast = K.function( [model_cl.input], [prelast_layer.output] )

# get center values
centers = cl_layer.get_weights()[0]
print ("centers.shape={}".format(centers.shape))
#assert (centers.shape==(cnt_classes,prelast_size)) # doesn't work if not pre-last layer

# load data
my_iterator = MyTfrecordIterator(tfrecord_path=tfrecord_filepath)
data_yielder = my_iterator.get_iterator_xy_ydummy()

# compute distance from center and log to file
for batch_id in range(my_iterator.len()):
#for batch_id in range(int(my_iterator.len() / 20)):  #full train set will take 4 hours
    if batch_id%100==0:
        print ("Batch {}/{}".format(batch_id,my_iterator.len()))
    # get data
    x_y,y_dummy = next(data_yielder)

    X = x_y[0]
    y = x_y[1]
    Y = np.argmax(y, axis=1)

    assert ( X.shape[1:4] == (256,256,3) )
    assert ( y.shape[1:2] == (cnt_classes,) )
    # calc dist to center
    prelast_output = func_prelast(x_y)[0]
    #assert (prelast_output.shape[1:2] == (prelast_size,)) # doesn't work if not pre-last layer
    dists_to_centers_batch = dists_to_center_f (centers = centers, prelast_activations=prelast_output, p=p_minkowski)
    assert (dists_to_centers_batch.shape[1:2] == (cnt_classes,))

    # normalize by full dataset's min/max
    dists_to_centers_batch = (dists_to_centers_batch - dist_min) / (dist_max-dist_min)

    # output 2 1D arrays to file: correct;dist
    y_ravel = np.ravel(y)
    #print ("np.sum(y_ravel): {}".format(np.sum(y_ravel)))
    #print ("dists_to_centers_batch.shape: {}".format(dists_to_centers_batch.shape))
    dist_ravel = np.ravel(dists_to_centers_batch)

    for sample_ind in range(X.shape[0]):

        Y_i = Y[sample_ind]
        if Y_i in selected_class_inds:

            #print ("    sample_ind: {}".format(sample_ind))
            #print ("y_ravel[sample_ind]:{}".format(y_ravel[sample_ind]))
            #print ("dist_ravel[sample_ind]:{}".format(dist_ravel[sample_ind]))
            #file_name = "{}_{}.jpg".format(  batch_id,sample_ind )
            #file_name = os.path.join(verif_result_dir, file_name )

            x_i = X[sample_ind,:,:,:]
            x_i = K.eval( x_i )
            x_i = np.uint8 ( x_i * 255 )
            x_i = Image.fromarray(x_i)

            #print ("type(x_i): {}".format(type(x_i)))
            #print ("x_i.shape:{}".format(x_i.shape))
            #x_i.save(file_name)

            dists_to_centers_sample = dists_to_centers_batch[sample_ind,:]
            assert (dists_to_centers_sample.shape==(194,))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.9, 2.6), width_ratios=[2, 1])
            fig.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.imshow( x_i )

            dists_to_centers_of_selected = np.array( [ dists_to_centers_sample[sel_class_ind] for sel_class_ind in selected_class_inds] )
            colors = [ "red" if sel_class_ind!=Y_i else "green" for sel_class_ind in selected_class_inds]
            ax2.hlines(dists_to_centers_of_selected, xmin=0.01, xmax=0.05, colors=colors)
            [ax2.text(0.1, dist_item-1e-2, "{:.3f} {}".format(dist_item, selected_class_names[i])) for (i,dist_item) in enumerate(dists_to_centers_of_selected) ]

            # eer
            ax2.hlines ([eer_threshold], xmin=0.01, xmax=1.0, colors=["blue"], linestyles='dashed')
            ax2.text (0.1, eer_threshold+1e-2, "Thr.@EER", color="blue")

            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_xlim(0.0, 1.0)
            ax2.set_ylim( np.min(dists_to_centers_of_selected)-0.1, np.max(dists_to_centers_of_selected)+0.1)
            for pos in ['right', 'top', 'bottom', 'left']:
                plt.gca().spines[pos].set_visible(False)

            file_name_numbers = "{}_{}_numbers.jpg".format(  batch_id,sample_ind )
            file_name_numbers = os.path.join(verif_result_dir, file_name_numbers )

            plt.savefig(file_name_numbers)
            plt.close()


            #plt.hlines ()


    df = pd.DataFrame ()

    m = X.shape[0]
    n = dists_to_centers_batch.shape[1]

    df['batch_sample'] = ["{}_{}".format(batch_id, sample_id) for sample_id in np.repeat(np.arange(m), n) ]
    df['correct'] = y_ravel
    df['dist'] = dist_ravel
    df.to_csv(dists_filename, mode="a", header=False, index=False)


