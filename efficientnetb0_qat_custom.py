# efficientnetb0_qat_custom.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config as qc
from tensorflow_model_optimization.quantization.keras import QuantizeConfig
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer, quantize_apply, quantize_annotate_model
import tensorflow_model_optimization as tfmot


class NoOpQuantizeConfig(QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


# Annotate unsupported layers with NoOpQuantizeConfig
def apply_custom_annotation(model):
    annotated_layers = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Multiply, tf.keras.layers.Rescaling, tf.keras.layers.Normalization, tf.keras.layers.BatchNormalization)):
            annotated_layers.append(quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig()))
        else:
            annotated_layers.append(layer)
    return tf.keras.Sequential(annotated_layers)


# Build EfficientNetB0 base model with annotations
input_tensor = Input(shape=(224, 224, 3))
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output)

# Apply annotation to unsupported layers
annotated_model = apply_custom_annotation(model)

# Apply QAT
quant_aware_model = quantize_apply(annotated_model)

# Compile
quant_aware_model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

# Data generators (replace 'dataset_path/' with actual path)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_size = 224
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=[0.8, 1.0],
    validation_split=0.1
)

train_gen = datagen.flow_from_directory(
    'dataset_path/',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'dataset_path/',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Train model
quant_aware_model.fit(train_gen,
                      validation_data=val_gen,
                      epochs=20)

# Save final model
quant_aware_model.save('efficientnetb0_qat_custom.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
tflite_model = converter.convert()

with open('efficientnetb0_qat_custom.tflite', 'wb') as f:
    f.write(tflite_model)

print("QAT complete and model saved as .h5 and .tflite")
