#!/usr/bin/python
# программа для сбора скринов экрана по команде
# сохранение идет в папку screens (переменная screen_path)
# Ctrl + Alt + W начать сбор
# Ctrl + Alt + Q выход

# библиотеки
import pyautogui
import keyboard
import time
import os

# переменные
screen_path = 'screens'

# функции
def value_i():
    '''
    value_i - функция возврата значения i при первом и повторном запусках
    '''
    screens_list = os.listdir(work_dir)
    if len(screens_list) == 0:
        i = 0
    else:
        screens_list = [int(i.replace('.png','')) for i in screens_list]
        i = max(screens_list)+1
    return i
    
def main_while():
    '''
    main_while - основной цикл, который делает скриншоты
    '''
    global i
    n = i
    print('Пуск.')
    while True:
        if (i != 0) and (keyboard.is_pressed('Ctrl + Alt + Q')):
            print(f'Остановка. Всего записано {i-n} файлов.')
            break
        pyautogui.screenshot(os.path.join(work_dir, str(i)+'.png'))
        i += 1
        time.sleep(0.2) # 0.02

# основной код

# записываем полный путь до папки со скринами
work_dir = os.path.join(os.getcwd(), screen_path)
# задаем значение i в зависимости от пред. максимального номера
if os.path.exists(work_dir):
    i = value_i()
else:
    os.mkdir(screen_path)
    i = value_i()
print('Готов к пуску.')
# по команде с клавиатуры записываем скрины
keyboard.add_hotkey('Ctrl + Alt + W', main_while)
# команда для выхода.
keyboard.wait('Ctrl + Alt + Q')
