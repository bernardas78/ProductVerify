from tensorflow.keras import backend as K
from proxyNcaLayer_Cosine import ProxyNcaLayer

layer = ProxyNcaLayer(Softmax_size=3, PreLastDense_size=2, alpha=0.5)

x = K.constant([[1,2],[-3,4],[5,-6],[-7,-8]])
y = K.constant( [ [1,0,0], [0,1,0], [0,0,1], [1,0,0] ])
res = layer( [x,y] )
print (res)