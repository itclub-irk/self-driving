#!/usr/bin/python3

### Импорт библиотек ###
import telebot
import os

from encoder import build_encoder, build_decoder, preprocessing_input_image
from encoder import get_result_ae, MSE

encoder = build_encoder('encoder_weights_10.h5')
decoder = build_decoder('decoder_weights_10.h5')
n = 0

# Файл, содержащий api-key
WORK_DIR = 'bot'
AUTH_FILE_NAME = os.path.join(WORK_DIR, 'auth_info.txt')
example_plain = open(os.path.join(WORK_DIR, 'example.jpg'), 'rb')

# Читаем файл с ключем
with open(AUTH_FILE_NAME, 'r', encoding='utf-8') as f:
    auth_info = f.readline()

# Создание бота
bot = telebot.TeleBot(auth_info)

@bot.message_handler(commands=['start'])
def help_mess(message):
    bot.send_photo(
        message.from_user.id,
        example_plain,
        caption='Отправь боту фото самолета (как на примере) или любую другую картинку.\nБот ответит есть самолет на этой картинке или его нет.'
    )

@bot.message_handler(content_types=['photo'])
def photo_mess(message):
    global n
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    new_file_name = f"{message.from_user.id}_{n}.jpg"
    with open(new_file_name, 'wb') as new_file:
        new_file.write(downloaded_file)
    n += 1
    input_array = preprocessing_input_image(new_file_name)
    latent_dims, result_array = get_result_ae(input_array, encoder, decoder)
    mse = MSE(result_array, input_array)
    print(mse)
    if mse <= 0.0035:
        bot.send_message(
            message.from_user.id,
            '<b>НОРМА</b>\nЭто самолет',
            parse_mode='html'
        )
    else:
        bot.send_message(
            message.from_user.id,
            '<b>АНОМАЛИЯ</b>\nЭто не самолет',
            parse_mode='html'
        )

# Непрерывное получение ботом сообщения с сервера
bot.polling(none_stop=True, interval=0)
