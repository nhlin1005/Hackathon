import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Rescaling, RandomFlip, RandomRotation, RandomZoom,
    RandomTranslation, RandomContrast, Conv2D, MaxPooling2D,
    AveragePooling2D, Dropout, Flatten, Dense, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

rescale = Rescaling(1./255)

data_augmentation = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.05),
    RandomTranslation(0.05, 0.05),
    RandomContrast(0.05)
])

data_dir = "C:/Users/Nanzh/PycharmProjects/pythonProject6/cnnapp/animals10-split-70-15-15/train"
val_dir = "C:/Users/Nanzh/PycharmProjects/pythonProject6/cnnapp/animals10-split-70-15-15/val"

image_size = (224, 224)
batch_size = 64

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir, image_size=image_size, batch_size=batch_size, shuffle=True
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=image_size, batch_size=batch_size, shuffle=True
)

class_names = train_dataset.class_names


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(lambda x, y: (rescale(x), y)).map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (rescale(x), y)).prefetch(AUTOTUNE)


def inception_module(x, f1, f3, f5, proj):
    path1 = Conv2D(f1, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    path2 = Conv2D(f3[0], (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    path2 = Conv2D(f3[1], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(path2)
    path3 = Conv2D(f5[0], (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    path3 = Conv2D(f5[1], (5,5), padding='same', activation='relu', kernel_initializer='he_normal')(path3)
    path4 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    path4 = Conv2D(proj, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(path4)
    return Concatenate(axis=-1)([path1, path2, path3, path4])


def build_googlenet(input_shape=(224,224,3), num_classes=10):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = Conv2D(64, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(192, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = inception_module(x, 64, [96,128], [16,32], 32)
    x = inception_module(x, 128, [128,192], [32,96], 64)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = inception_module(x, 192, [96,208], [16,48], 64)
    x = inception_module(x, 160, [112,224], [24,64], 64)
    x = inception_module(x, 128, [128,256], [24,64], 64)
    x = inception_module(x, 112, [144,288], [32,64], 64)
    x = inception_module(x, 256, [160,320], [32,128], 128)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = inception_module(x, 256, [160,320], [32,128], 128)
    x = inception_module(x, 384, [192,384], [48,128], 128)

    x = AveragePooling2D((7,7), strides=(1,1), padding='valid')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='linear', kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


model = build_googlenet(input_shape=(224,224,3), num_classes=10)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=40,
    callbacks=[early_stop, reduce_lr]
)


model.save("googlenet_animals10_final_40epochs.h5")


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()