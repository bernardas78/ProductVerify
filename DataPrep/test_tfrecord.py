import os
from Globals.globalvars import MyTfrecordIterator, MyPairsIterator,MyTripletIterator
from Globals.globalvars import Glb
import random
import time

#import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

tfrecord_fullds_path = os.path.join(Glb.images_folder, "PV_TFRecord", "Train10.tfrecords")
tfrecords_byclass_path = os.path.join(Glb.images_folder, "PV_TFRecord_ByClass", "Train10")

train_iterator = MyTripletIterator(tfrecord_fullds_path=tfrecord_fullds_path, tfrecords_byclass_path=tfrecords_byclass_path)
#train_iterator = MyPairsIterator(tfrecord_fullds_path=tfrecord_fullds_path, tfrecords_byclass_path=tfrecords_byclass_path)
#train_iterator = MyTfrecordIterator(tfrecord_path=tfrecord_fullds_path)

i=0
now= time.time()
print ("train_iterator.len:{}".format(train_iterator.len()))

for x,y in train_iterator.get_triplets_iterator():
#for x,y in train_iterator.get_iterator_pair():
#for x1 in train_iterator.get_iterator_xy_ydummy():
    a,p,n=x
    #dist_pos, dist_neg=y
    print (a.shape)
    print (p.shape)
    print (n.shape)
    #print(np.argmax(y,axis=1))
    #x1,x2=x
    #print("dist_pos:{}".format(dist_pos))
    #print("dist_neg.shape:{}".format(dist_neg.shape))

    minibatch_size = a.shape[0]
    for i in range(a.shape[0]):
        f, axarr = plt.subplots(1, 3 )
        axarr[0].imshow(a[i,:,:,:])
        axarr[1].imshow(p[i,:,:,:])
        axarr[2].imshow(n[i,:,:,:])
        plt.savefig("mygraph{}.png".format(i))
        plt.close()

    #if (i%100==0):
    #    print(i)
    #if i>=train_iterator.len():
    #    break
    #if i>10:
    #    break
    break

#print (x1.shape)
#print (x2.shape)
#print (y.shape)
#print(y)
print ("Time elapsed: {}sec".format(time.time()-now))