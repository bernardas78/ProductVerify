import os
from Globals.globalvars import MyTfrecordIterator, MyPairsIterator
import random
import time

#import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

tfrecords_byclass_path = os.path.join ( r"A:\IsKnown_Images\PV_TFRecord_ByClass\Train10" )
tfrecord_fullds_path = os.path.join ( r"A:\IsKnown_Images\PV_TFRecord\Train10.tfrecords" )

train_iterator = MyPairsIterator(tfrecord_fullds_path=tfrecord_fullds_path, tfrecords_byclass_path=tfrecords_byclass_path)
#train_iterator = MyTfrecordIterator(tfrecord_path=tfrecord_fullds_path)

i=0
now= time.time()
print ("train_iterator.len:{}".format(train_iterator.len()))

for x,y in train_iterator.get_iterator_pair():
#for x1 in train_iterator.get_iterator_xy_ydummy():
    #print (x1.shape)
    #print (x2.shape)
    #print (y.shape)
    #print(np.argmax(y,axis=1))
    x1,x2=x
    print(y)

    f, axarr = plt.subplots(1, 2)

    rand_ind =random.randint(0,x1.shape[0]-1)
    axarr[0].imshow(x1[rand_ind,:,:,:])
    axarr[1].imshow(x2[rand_ind,:,:,:])

    print ("rand_ind:{}".format(rand_ind))
    plt.savefig("mygraph{}.png".format(i))

    i+=1
    if (i%100==0):
        print(i)
    if i>=train_iterator.len():
        break
    if i>10:
        break
    #break

#print (x1.shape)
#print (x2.shape)
#print (y.shape)
#print(y)
print ("Time elapsed: {}sec".format(time.time()-now))