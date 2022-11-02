import tensorflow as tf

feature_dict = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width' : tf.io.FixedLenFeature([], tf.int64),
      'lbl' : tf.io.FixedLenFeature([], tf.int64),
      'img' : tf.io.FixedLenFeature([], tf.string)
    }

def parser(data_record):
    ''' TFRecord parser '''

    #feature_dict = {
    #  'height': tf.io.FixedLenFeature([], tf.int64),
    #  'width' : tf.io.FixedLenFeature([], tf.int64),
    #  'lbl' : tf.io.FixedLenFeature([], tf.int64),
    #  'img' : tf.io.FixedLenFeature([], tf.string)
    #}
    sample = tf.io.parse_single_example(data_record, feature_dict)
    label = tf.cast(sample['lbl'], tf.int32)

    h = tf.cast(sample['height'], tf.int32)
    w = tf.cast(sample['width'], tf.int32)
    image = tf.io.decode_image(sample['img'], channels=3)
    image = tf.reshape(image,[h,w,3])

    return image, label