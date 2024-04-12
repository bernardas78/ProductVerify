from tensorflow.keras import backend as K
from proxyNcaLayer_Mink import ProxyNcaLayer

layer = ProxyNcaLayer(Softmax_size=3, PreLastDense_size=2, alpha=0.5, p=2)
layer.get_config()
print ("proxynca_layer.centers BFR: {}".format(layer.get_weights()))
#layer.centers = [ [0.,1.], [-1.,-1.], [1.,1.]]
#layer.get_config()
#print ("proxynca_layer.centers AFT: {}".format(layer.centers))

#x = K.constant([[1,2],[-3,4],[5,-6],[-7,-8]])
#y = K.constant( [ [1,0,0], [0,1,0], [0,0,1], [1,0,0] ])

x = K.constant([[0,1]])
y = K.constant( [ [1,0,0] ])

res = layer( [x,y] )
print (res)