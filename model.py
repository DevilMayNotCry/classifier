import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFViTModel, ViTConfig, TFViTForImageClassification

configuration = ViTConfig(hidden_size = 128,
                          num_hidden_layers = 8,
                          num_attention_heads = 8,
                          intermediate_size = 128,
                          hidden_act = "gelu")

base_model = TFViTForImageClassification(configuration)

pixel_values = tf.keras.layers.Input(shape=(3,224,224), name='pixel_values', dtype='float32')

vit = base_model.vit(pixel_values)[0]
x = tf.keras.layers.Dense(64, activation='relu')(vit[:, 0, :])
classifier = tf.keras.layers.Dense(6, activation='softmax', name='outputs')(x)

keras_model = tf.keras.Model(inputs=pixel_values, outputs=classifier)
