import numpy as np
import math
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import os
import pandas as pd
import sys
from trained_model_names import model_names, experSuffix_names
from PIL import Image, ImageDraw,ImageFont

sys.path.insert(0,'..')

from CenterLoss.centerLossLayer import CenterLossLayer, center_loss
from Globals.globalvars import Glb, MyTfrecordIterator

if Glb.isRetellect:
    data_dir = Glb.images_balanced_folder_retellect
    tfrecord_dir = os.path.join(data_dir, "PV_TFRecord")
    cnt_classes = 15
else:
    data_dir = Glb.images_balanced_folder
    tfrecord_dir = os.path.join(Glb.images_folder, "PV_TFRecord")
    cnt_classes = 194

set_name = "Unknown"
#set_name = "Train10"

experName = ""
preClIndex = -1234
prelast_size = -5423
distName = "Eucl"
p_minkowski = -1
inclInterCenter = False
lambda2 = -1234234.34543
lambda1 = "n/a"

if sys.argv[1] == "lambda1" or sys.argv[1] == "cosineCl":
    experName = sys.argv[1]
    lambda1 = sys.argv[2]
elif sys.argv[1] == "preClIndex":
    experName = sys.argv[1]
    preClIndex = sys.argv[2]
elif sys.argv[1] == "retellect":
    experName = sys.argv[1]
    prelast_size = 512
else:
    prelast_size = int(sys.argv[1]) #512
    distName = sys.argv[2] #"Eucl"
    p_minkowski = int(sys.argv[3]) #2
    inclInterCenter = sys.argv[4]=="True"
    lambda2 = float(sys.argv[5])


mink_suffix = "_{}".format(p_minkowski) if distName == "Minkowski" else ""
interc_suffix = "_{:.3f}".format(lambda2) if inclInterCenter else ""

tfrecord_filepath = os.path.join(tfrecord_dir, "{}.tfrecords".format(set_name))

#dists_filename = os.path.join ( Glb.results_folder, "Dists", "dists_{}_{}{}_{}{}.csv".format(prelast_size,distName,mink_suffix,inclInterCenter,interc_suffix) )
experSuffix = experSuffix_names(distName,prelast_size,p_minkowski,inclInterCenter,lambda2,experName,preClIndex, lambda1)
dists_filename = os.path.join ( Glb.results_folder, "Dists", "dists_top1{}csv".format(experSuffix) )
df = pd.DataFrame (columns = ["correct", "dist"] )
df.to_csv(dists_filename, mode="w", header=True, index=False)

prod_names = pd.read_csv(r"D:\Retellect\sco-vision-service-api\model_20220629_15prekes.csv", header=None).iloc[:,0].tolist()

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
assert (centers.shape==(cnt_classes,prelast_size))

# load data
my_iterator = MyTfrecordIterator(tfrecord_path=tfrecord_filepath, cnt_classes=cnt_classes)
data_yielder = my_iterator.get_iterator_xy_ydummy()

#print(model_cl.summary())

# compute distance from center and log to file
for batch_id in range(my_iterator.len()):
#for batch_id in range(int(my_iterator.len() / 20)):  #full train set will take 4 hours
    if batch_id%100==0:
        print ("Batch {}/{}".format(batch_id,my_iterator.len()))
    # get data
    x_y,y_dummy = next(data_yielder)
    assert ( x_y[0].shape[1:4] == (256,256,3) )
    assert ( x_y[1].shape[1:2] == (cnt_classes,) )

    #predict top 1
    y_hat, _ = model_cl.predict(x_y)
    Y_hat = np.argmax(y_hat, axis=1)
    #print ("y_hat:{}".format(Y_hat))

# calc dist to center
    prelast_output = func_prelast(x_y)[0]
    assert (prelast_output.shape[1:2] == (prelast_size,))
    dists_to_centers_batch = dists_to_center_f (centers = centers, prelast_activations=prelast_output, p=p_minkowski)
    assert (dists_to_centers_batch.shape[1:2] == (cnt_classes,))

# filter dists of top1's center
    batch_size = y_hat.shape[0]
    #print ("batch_size:{}".format(batch_size))
    dists_to_top1_centers_batch = dists_to_centers_batch[np.arange(batch_size),Y_hat]
    #print ("dists_to_centers_batch.shape: {}".format(dists_to_centers_batch))
    #print("Y_hat:{}".format(Y_hat))
    #print("dists_to_top1_centers_batch.shape: {}".format(dists_to_top1_centers_batch))

# output 2 1D arrays to file: correct;dist
    y_ravel = np.zeros(batch_size)
    dist_ravel = dists_to_top1_centers_batch
    df = pd.DataFrame ()
    df['correct'] = y_ravel
    df['dist'] = dist_ravel
    df.to_csv(dists_filename, mode="a", header=False, index=False)

    for img_id in range(batch_size):
        #print ("x_y[0].shape: {}".format(x_y[0][img_id,:,:,:].shape))
        img = Image.fromarray ( np.uint8(K.eval(x_y[0][img_id,:,:,:])*255.), mode="RGB" )

        I1 = ImageDraw.Draw(img)
        font=ImageFont.truetype("arial",size=20)
        I1.text((28, 36), "Top 1: {}".format(prod_names[Y_hat[img_id]]), fill=(0, 255, 0), font=font)
        I1.text((28, 56), "Dist: {}".format(dists_to_top1_centers_batch[img_id]), fill=(0, 255, 0), font=font)

        img_filename = os.path.join(r"C:\Retellect.Demo20230726.Balanced\Unknown.Dist", "{}_{}.jpg".format(batch_id,img_id))
        img.save(img_filename)

# sanity check
print ("Sanity check: dists calculated using [prelast_output-centers] vs. centerloss")
_, cl = model_cl.predict(x_y)
y = x_y[1]
for i in range( y.shape[0] ):
#for i in range(1):
    Y = np.argmax(y[i])
    dist_center = prelast_output[i] - centers[Y]
    #print("dist_center:{}".format(dist_center))

    if distName=="Eucl":
        dist_center_total = np.sqrt ( np.sum(dist_center**2) )
    elif distName=="Manhattan":
        dist_center_total = np.sum ( np.abs(dist_center) )
    elif distName == "Minkowski":
        dist_center_total = np.power(np.sum(np.power(np.abs(dist_center),p_minkowski)),1/p_minkowski)
    else:
        raise Exception("Unknown dist name")

    print ( "dist_center_total:{}; cl:{}".format(dist_center_total, cl[i][0]))
    assert math.isclose(dist_center_total, cl[i][0], rel_tol=1e-4)

