import numpy as np
import math
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import os
from ProxyNCA import proxyNcaLayer
from CenterLoss import centerLossLayer_Eucl

from Globals.globalvars import Glb

model_cl_date = "20240221"
uq_num = "65984501"

model_proxynca_filename = os.path.join(Glb.results_folder, "Models", "model_proxynca_{}_{}.h5".format (model_cl_date, uq_num) )
print ("Loading {}".format(model_proxynca_filename))
model_proxynca = load_model(model_proxynca_filename, custom_objects={'ProxyNcaLayer': proxyNcaLayer.ProxyNcaLayer, 'proxynca_loss': proxyNcaLayer.proxynca_loss})
#model_proxynca = load_model(model_proxynca_filename, custom_objects={'dfdaf': centerLossLayer_Eucl.CenterLossLayer,
#                                                                     'fdafd': centerLossLayer_Eucl.center_loss})
print ("Loaded")

# get center values
proxynca_layer = model_proxynca.get_layer('proxyncalayer')
centers = proxynca_layer.get_weights()[0]
print ("centers.shape: {}".format(centers.shape))
assert (centers.shape==(194,512))


# random data
x = np.random.randn(2,256,256,3)

y = np.zeros([2,194])
y[0,0]=1
y[1,1]=1

inp = [x,y]


#prelast output
prelast_layer = model_proxynca.get_layer('Dense_post_originally_prelast')
func_prelast = K.function( [model_proxynca.input], [prelast_layer.output] )
prelast_output = func_prelast(inp)[0]
assert (prelast_output.shape==(2,512))

#centerloss output
_, cl = model_proxynca.predict(inp)


#eucl distance from prelast - compare to centerloss
for i in range(2):
    Y = np.argmax(y[i])
    dist_center = prelast_output[i] - centers[Y]
    dist_center_sqsum = np.sum ( np.dot(dist_center,dist_center) )
    print ( "dist_center_sqsum:{}; cl:{}".format(dist_center_sqsum, cl[i][0]))
    assert math.isclose(dist_center_sqsum, cl[i][0], rel_tol=1e-5)