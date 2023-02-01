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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Reshape, Dense, Flatten
import numpy as np
from PIL import Image

def build_encoder(encoder_weights:str = 'encoder_weights_2.h5'):
    '''
    Создание энкодера и загрузка в него весов
    вход:
        encoder_weights - путь к весам энкодера
    выход:
        модель энкодера
    '''
    enc_in = Input((112,112,3))
    enc_1 = Conv2D(32, (3,3), padding='same')(enc_in)
    enc_1 = Conv2D(32, (3,3), padding='same')(enc_1)
    enc_1 = MaxPooling2D((2,2))(enc_1)
    enc_2 = Conv2D(64, (3,3), padding='same')(enc_1)
    enc_2 = Conv2D(64, (3,3), padding='same')(enc_2)
    enc_2 = MaxPooling2D((2,2))(enc_2)
    enc_3 = Conv2D(128, (3,3), padding='same')(enc_2)
    enc_3 = Conv2D(128, (3,3), padding='same')(enc_3)
    enc_3 = MaxPooling2D((2,2))(enc_3)
    enc_4 = Conv2D(256, (3,3), padding='same')(enc_3)
    enc_4 = Conv2D(256, (3,3), padding='same')(enc_4)
    enc_4 = MaxPooling2D((2,2))(enc_4)
    enc_5 = Flatten()(enc_4)
    enc_5 = Dense(4096)(enc_5)
    enc_5 = Dense(1024)(enc_5)
    encoder = Model(enc_in, enc_5)
    encoder.load_weights(encoder_weights)
    return encoder

def build_decoder(decoder_weights:str = 'decoder_weights_2.h5'):
    '''
    Создание декодера и загрузка в него весов
    вход:
        decoder_weights - путь к весам декодера
    выход:
        модель декодера
    '''
    dec_in = Input((1024,))
    dec_1 = Dense(4096)(dec_in)
    dec_1 = Dense(7*7*256)(dec_1)
    dec_1 = Reshape((7,7,256))(dec_1)
    dec_2 = UpSampling2D((2,2))(dec_1)
    dec_2 = Conv2D(256, (3,3), padding='same')(dec_2)
    dec_2 = Conv2D(256, (3,3), padding='same')(dec_2)
    dec_3 = UpSampling2D((2,2))(dec_2)
    dec_3 = Conv2D(128, (3,3), padding='same')(dec_3)
    dec_3 = Conv2D(128, (3,3), padding='same')(dec_3)
    dec_4 = UpSampling2D((2,2))(dec_3)
    dec_4 = Conv2D(64, (3,3), padding='same')(dec_4)
    dec_4 = Conv2D(64, (3,3), padding='same')(dec_4)
    dec_5 = UpSampling2D((2,2))(dec_4)
    dec_5 = Conv2D(32, (3,3), padding='same')(dec_5)
    dec_5 = Conv2D(32, (3,3), padding='same')(dec_5)
    dec_6 = Conv2D(3, (1,1), padding='same', activation='relu')(dec_5)
    decoder = Model(dec_in, dec_6)
    decoder.load_weights(decoder_weights)
    return decoder

def preprocessing_input_image(path_image:str, mode='RGB', scaler=0.99/255):
    '''
    Предподготовка файла изображения 
    вход:
        encoder_weights - путь к весам энкодера
    выход:
        numpy-массив размера (112,112,3) или (112,112,1) в зависимости от режима
    '''
    path_image = os.path.join(os.getcwd(), path_image)
    input_image = Image.open(path_image)
    old_size = np.array(input_image.size)
    new_size = old_size * (112/old_size.min())
    new_size = np.rint(new_size).astype('int')
    input_image = input_image.resize(size=tuple(new_size))
    new_image = Image.new(mode=mode, size=(112,112), color=0)
    new_image.paste(input_image, (0,0))
    output_arr = np.array(new_image) * scaler
    if mode == 'L':
        output_arr = output_arr[:, :, np.newaxis]
    output_arr = output_arr[np.newaxis, :, :, :]
    return output_arr

def get_result_ae(input_array, encoder, decoder):
    latent_dims = encoder(input_array).numpy()
    predict_ae = decoder(latent_dims).numpy()
    return (latent_dims, predict_ae)

def MSE(y_pred, y_true):
    result = y_pred - y_true
    result = result * result
    return result.mean()


if __name__ == "__main__":
    base_folder_path = os.path.join(os.getcwd(), 'base/for_ae/Valid/plane')
    image_name = os.listdir(base_folder_path)
    image_name.sort()
    encoder = build_encoder('encoder_weights_5.h5')
    decoder = build_decoder('decoder_weights_5.h5')

    l = []
    for i in image_name:
        input_array = preprocessing_input_image(os.path.join(base_folder_path, i))
        latent_dims, result_array = get_result_ae(input_array, encoder, decoder)
        mse = MSE(result_array, input_array)
        l.append(mse)
        if mse <= 0.02:
            print(i, 'САМОЛЕТ', mse)
        else:
            print(i, 'не самолет', mse)
    
