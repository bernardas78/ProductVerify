import tensorflow as tf
from Globals.globalvars import Glb_Iterators, Glb
import os
from PIL import Image
import numpy as np
import base64
from tfrecord_reader import parser
import time
import random

set_name="Train"
#set_name="Train10"
#set_name="Val"
#set_name="Test"

batch_size=32
div255_resnet = "div255"


img_filepath = os.path.join( Glb.images_balanced_folder, set_name)
tfrecord_path = os.path.join ( Glb.images_folder, "PV_TFRecord", "{}.tfrecords".format(set_name) )

#img_filepath = os.path.join( "C:\\Retellect.Demo20230726.Balanced", set_name)
#tfrecord_path = os.path.join ( "C:\\Retellect.Demo20230726.Balanced", "{}.tfrecords".format(set_name) )

img_filepath = os.path.join( "A:/Fruits360/Balanced", set_name)
tfrecord_path = os.path.join ( Glb.images_folder, "PV_TFRecord_Fruits360", "{}.tfrecords".format(set_name) )



def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytesS_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _floatS_feature(values):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# all file names to list
allfiles_path = []
for barcode_path in os.listdir(img_filepath):
    allfiles_path += [ os.path.join(img_filepath,barcode_path,filepath) for filepath in os.listdir( os.path.join (img_filepath,barcode_path )) ]
random.shuffle(allfiles_path)

barcodes = os.listdir(img_filepath)

now= time.time()

j=0
with tf.io.TFRecordWriter(tfrecord_path) as file_writer:
    for filename in allfiles_path:
        with open(filename, 'rb') as f:

            in_jpg_encoding = tf.io.read_file(filename)

            image_shape = tf.io.extract_jpeg_shape(in_jpg_encoding, output_type=tf.dtypes.int32, name=None)

            # Fruits360 original images were 100x100x3
            if not np.array_equal ( image_shape, [256,256,3] ):
                #print ("resize")
                image = tf.image.decode_jpeg(in_jpg_encoding, channels=3)
                image = tf.cast(tf.image.resize(image, [256, 256]), tf.uint8)
                in_jpg_encoding = tf.image.encode_jpeg( image )
                image_shape = tf.io.extract_jpeg_shape(in_jpg_encoding, output_type=tf.dtypes.int32, name=None)

            #print (image_shape)

            # img = Image.open(filename)
            # image_raw = img.tostring()
            feature = {
                'height': _int64_feature(image_shape[0]),
                'width': _int64_feature(image_shape[1]),
                'lbl': _int64_feature( barcodes.index(filename.split('\\')[-2]) ),
                'img': _bytes_feature(in_jpg_encoding)
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            example_serialized = example_proto.SerializeToString()

            file_writer.write(example_serialized)

            #j+=1
            #if j>=20:
            #    break

#print ("Time elapsed: {}sec".format(time.time()-now))



# Retrieval



now = time.time()

dataset = tf.data.TFRecordDataset(tfrecord_path).map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#raw_dataset = tf.data.TFRecordDataset([dest_path]).batch(32)

i=0
for raw_batch in dataset.batch(32):
    #print ("batch {}".format(i))
    i+=1
    #print(repr(raw_batch))
    #example_proto = tf.train.Example.FromString(repr(raw_batch[0]))
print ("Time elapsed: {}sec".format(time.time()-now))
print ("raw_batch[0].shape: {}".format(raw_batch[0].shape))
print ("raw_batch[1]: {}".format(raw_batch[1]))
#Time elapsed: 5.329460859298706sec