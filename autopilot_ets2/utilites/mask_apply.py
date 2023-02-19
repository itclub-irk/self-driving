#!/usr/bin/python3

from PIL import Image
import numpy as np
from os import path

# имя изображения для наложения маски
img_1_name = '1195.png'

# папки с реальными скринами и с масками
folder_img = 'Dataset/Roads/Before'
folder_msk = 'Dataset/Roads/After/'

# открытие изображения
img_1 = Image.open(path.join(folder_img, img_1_name))

# открытие маски и добавление ей альфа-канала
alpha_chenal = 100
msk_1 = np.array(Image.open(path.join(folder_msk, img_1_name)))
alpha = np.full((msk_1.shape[0], msk_1.shape[1], 1), alpha_chenal).astype('uint8')
msk_1 = np.concatenate((msk_1, alpha), axis=-1)
msk_1 = Image.fromarray(msk_1, mode='RGBA')

# наложение маски
img_1.paste(msk_1, (0,0), msk_1)

# сохранение
img_1.save('example_' + img_1_name)

# освобождаем память
img_1.close()
