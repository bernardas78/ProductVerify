import numpy as np
import math
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import os
from CenterLoss import centerLossLayer

from Globals.globalvars import Glb

model_cl_date = "20221017"

model_cl_filename = os.path.join(Glb.results_folder, "model_centerloss_{}.h5".format (model_cl_date) )
print ("Loading {}".format(model_cl_filename))
model_cl = load_model(model_cl_filename, custom_objects={'CenterLossLayer': centerLossLayer.CenterLossLayer, 'center_loss': centerLossLayer.center_loss})
print ("Loaded")

# get center values
cl_layer = model_cl.get_layer('centerlosslayer')
centers = cl_layer.get_weights()[0]
assert (centers.shape==(194,128))


# random data
x = np.random.randn(2,256,256,3)

y = np.zeros([2,194])
y[0,0]=1
y[1,1]=1

inp = [x,y]


#prelast output
prelast_layer = model_cl.get_layer('activation_8')
func_prelast = K.function( [model_cl.input], [prelast_layer.output] )
prelast_output = func_prelast(inp)[0]
assert (prelast_output.shape==(2,128))

#centerloss output
_, cl = model_cl.predict(inp)


#eucl distance from prelast - compare to centerloss
for i in range(2):
    Y = np.argmax(y[i])
    dist_center = prelast_output[i] - centers[Y]
    dist_center_sqsum = np.sum ( np.dot(dist_center,dist_center) )
    print ( "dist_center_sqsum:{}; cl:{}".format(dist_center_sqsum, cl[i][0]))
    assert math.isclose(dist_center_sqsum, cl[i][0], rel_tol=1e-5)