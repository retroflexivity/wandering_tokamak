# бот, постящий в канал 

import telebot
# import sched, time
import threading
from generator import Generator


token = open('tgbot_token.txt').read()

# вот этот — https://t.me/wandering_tokamak
chat_id = '@wandering_tokamak'
bot = telebot.TeleBot(token)

# функция, отправляющая в канал
def send_to_channel():
    print('started')
    # пытаемся пока не получится
    while True: 
        try:
            # генерируем картинку
            gen = Generator()
            image = gen.generate()
            print('generated')

            # отправляем в канал
            bot.send_photo(chat_id=chat_id, photo=image)
            print('sent')
            break
        except Exception as e:
            print(e)
            # raise e
    
    # спим...
    print('sleeping...')
    threading.Timer(600.0, send_to_channel).start()

send_to_channel()