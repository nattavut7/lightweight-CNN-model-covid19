# efficientnetb0_qat.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot
import os

# SETTINGS
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PRUNING = 50
EPOCHS_QAT = 10
NUM_CLASSES = 3  # COVID-19, Pneumonia, Normal
DATASET_PATH = 'dataset_path/'  # Change to your dataset path

# DATA PREPROCESSING
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=[0.8, 1.0],
    validation_split=0.1
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# BASE MODEL
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# PRUNING
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.98,
        begin_step=0,
        end_step=len(train_gen) * EPOCHS_PRUNING
    )
}

pruned_model = prune_low_magnitude(model, **pruning_params)

# COMPILE AND TRAIN PRUNED MODEL
pruned_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

pruned_model.fit(train_gen,
                 validation_data=val_gen,
                 epochs=EPOCHS_PRUNING,
                 callbacks=callbacks)

# STRIP PRUNING
stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

# QUANTIZATION-AWARE TRAINING (QAT)
quantize_model = tfmot.quantization.keras.quantize_model
qat_model = quantize_model(stripped_model)

qat_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

qat_model.fit(train_gen,
              validation_data=val_gen,
              epochs=EPOCHS_QAT)

# SAVE FINAL MODEL
qat_model.save('efficientnetb0_covid_qat_model.h5')

# CONVERT TO TFLITE
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
tflite_model = converter.convert()

with open('efficientnetb0_covid_qat_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Training complete and model saved as .h5 and .tflite")
