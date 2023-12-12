import tensorflow as tf
from tensorflow.keras import layers, models
from envs import *

"""define the input convolutional network"""
def make_embeding() -> models.Model:
  input = layers.Input(shape=IMAGE_SHAPE, batch_size=BATCH_SIZE, name='input_image')

  layer = layers.Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', activation='relu', name='conv1')(input)
  layer = layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(layer)
  layer = layers.BatchNormalization(name='batchnorm1')(layer)

  layer = layers.Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='same', activation='relu', name='conv2a')(layer)
  layer = layers.Conv2D(filters=96, kernel_size=3, strides=(1, 1), padding='same', activation='relu', name='conv2b')(layer)
  layer = layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(layer)
  layer = layers.BatchNormalization(name='batchnorm2')(layer)

  layer = layers.Conv2D(filters=96, kernel_size=1, strides=(1, 1), padding='same', activation='relu', name='conv3a')(layer)
  layer = layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', activation='relu', name='conv3b')(layer)
  layer = layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(layer)

  layer = layers.Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same', activation='relu', name='conv4a')(layer)
  layer = layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', activation='relu', name='conv4b')(layer)

  layer = layers.Conv2D(filters=256, kernel_size=1, strides=(1, 1), padding='same', activation='relu', name='conv5a')(layer)
  layer = layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', activation='relu', name='conv5b')(layer)

  layer = layers.Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same', activation='relu', name='conv6a')(layer)
  layer = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', activation='relu', name='conv6b')(layer)
  layer = layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling6')(layer)

  layer = layers.Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='same', activation='relu', name='conv7a')(layer)
  layer = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', activation='relu', name='conv7b')(layer)
  layer = layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling7')(layer)

  layer = layers.Flatten(name='flatten')(layer)
  output = layers.Dense(units=1024, activation='relu', name='dense')(layer)

  return models.Model(inputs=input, outputs=output, name='embeding_model')

"""make the siamese model"""
def make_model() -> models.Model:
  anchor_img = layers.Input(shape=IMAGE_SHAPE, batch_size=BATCH_SIZE, name='anchor_img')
  verification_img = layers.Input(shape=IMAGE_SHAPE, batch_size=BATCH_SIZE, name='verification_img')

  embeding = make_embeding()

  anchor_embeding, verification_embeding = embeding(anchor_img), embeding(verification_img)

  # distance layer
  Distance = layers.Lambda(lambda pair: tf.abs(pair[0] - pair[1]), name='distance')
  distance = Distance([anchor_embeding, verification_embeding])

  dense = layers.Dense(units=128, activation='relu', name='dense_distance')(distance)

  output = layers.Dense(units=1, activation='sigmoid', name='classifier')(dense)

  return models.Model(inputs=[anchor_img, verification_img], outputs=output, name='faceid_model')
