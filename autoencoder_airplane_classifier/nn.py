#!/usr/bin/python3

# отключение логирования
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# подключение GPU
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten
from tensorflow.keras.layers import UpSampling2D, Reshape
from tensorflow.keras.layers import Multiply, AveragePooling2D
from tensorflow.keras.layers import Concatenate, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

base = os.path.join(os.getcwd(), 'new_base')
train_path = base + '/train'


#Генератор изображений
datagen_t = ImageDataGenerator(
	rescale= 0.99/255, #Значения цвета меняем на дробные показания
	rotation_range=20, #Поворачиваем изображения при генерации выборки
	width_shift_range=0.1, #Двигаем изображения по ширине при генерации выборки
	height_shift_range=0.1, #Двигаем изображения по высоте при генерации выборки
	zoom_range=0.1, #Зумируем изображения при генерации выборки
	horizontal_flip=True, # Включаем отзеркаливание изображений
	fill_mode='nearest', #Заполнение пикселей вне границ ввода
	validation_split=0.2 #Указываем разделение изображений на обучающую и тестовую выборку
)

# обучающая выборка
img_gen_train = datagen_t.flow_from_directory(
	base + '/train', #Путь ко всей выборке
	target_size=(112, 112), #Размер изображений
	color_mode='rgb',
	batch_size=4, #Размер batch_size
	class_mode='input',
	shuffle=True, #Перемешивание выборки
	subset='training' # устанавливаем как набор для обучения
)

img_gen_valid = datagen_t.flow_from_directory(
	base + '/train', #Путь ко всей выборке
	target_size=(112, 112), #Размер изображений
	color_mode='rgb',
	batch_size=4, #Размер batch_size
	class_mode='input',
	shuffle=True, #Перемешивание выборки
	subset='validation' # устанавливаем как набор для обучения
)



enc_in = Input((112,112,3))

# Блок 1
enc_1 = Conv2D(32, (3,3), padding='same')(enc_in)
enc_1 = Conv2D(32, (3,3), padding='same')(enc_1)
enc_1 = MaxPooling2D((2,2))(enc_1)
# 56 x 56 x 32

# Блок 2
enc_2 = Conv2D(64, (3,3), padding='same')(enc_1)
enc_2 = Conv2D(64, (3,3), padding='same')(enc_2)
enc_2 = MaxPooling2D((2,2))(enc_2)
# 28 x 28 x 64

# Блок 3
enc_3 = Conv2D(128, (3,3), padding='same')(enc_2)
enc_3 = Conv2D(128, (3,3), padding='same')(enc_3)
enc_3 = MaxPooling2D((2,2))(enc_3)
# 14 x 14 x 128

# Блок 4
enc_4 = Conv2D(256, (3,3), padding='same')(enc_3)
enc_4 = Conv2D(256, (3,3), padding='same')(enc_4)
enc_4 = MaxPooling2D((2,2))(enc_4)
# 7 x 7 x 256

enc_5 = Flatten()(enc_4)
enc_5 = Dense(4096)(enc_5)
enc_5 = Dense(1024)(enc_5)

enc = Model(enc_in, enc_5)



dec_in = Input((1024,))
dec_1 = Dense(4096)(dec_in)
dec_1 = Dense(7*7*256)(dec_1)
dec_1 = Reshape((7,7,256))(dec_1)

dec_2 = UpSampling2D((2,2))(dec_1)
dec_2 = Conv2D(256, (3,3), padding='same')(dec_2)
dec_2 = Conv2D(256, (3,3), padding='same')(dec_2)
# 14 x 14 x 128

dec_3 = UpSampling2D((2,2))(dec_2)
dec_3 = Conv2D(128, (3,3), padding='same')(dec_3)
dec_3 = Conv2D(128, (3,3), padding='same')(dec_3)
# 28 x 28 x 64

dec_4 = UpSampling2D((2,2))(dec_3)
dec_4 = Conv2D(64, (3,3), padding='same')(dec_4)
dec_4 = Conv2D(64, (3,3), padding='same')(dec_4)
# 56 x 56 x 32

dec_5 = UpSampling2D((2,2))(dec_4)
dec_5 = Conv2D(32, (3,3), padding='same')(dec_5)
dec_5 = Conv2D(32, (3,3), padding='same')(dec_5)
# 112 x 112 x 16

dec_6 = Conv2D(3, (1,1), padding='same', activation='relu')(dec_5)
# 112 x 112 x 1

dec = Model(dec_in, dec_6)

ae = Model(enc_in, dec(enc(enc_in)))

model_filename = 'test_10.h5'
ae.load_weights(model_filename)

ae.compile(loss='mse',
          optimizer=Adam(0.0001),
          metrics=['accuracy']
)


m_ch = ModelCheckpoint(
	model_filename,
	monitor="val_loss",
	verbose=1,
	save_best_only=True,
	save_weights_only=True,
	mode="min",
	save_freq="epoch"
)

enc.save_weights('encoder_weights_10.h5')
dec.save_weights('decoder_weights_10.h5')
