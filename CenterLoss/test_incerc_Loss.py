import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import centerLossLayer_Eucl_InterC as cc

C=tf.constant ( np.array([[3,4],[-1.,-0]], dtype='float32')  )

cc.interc_delta (C, 0.5)


# load test
C = tf.random.normal((2, 2)) * 10
for i in range(1000):
    I=cc.interc_delta (C, 0.5)
    C=C+I
    if (i==0 or i==999):
        print (C)


# many centers test
C = tf.random.normal((194, 128)) * 10
for i in range(50):
    I=cc.interc_delta (C, 0.5)
    C=C+I
    print (np.sum(np.abs(I)))
