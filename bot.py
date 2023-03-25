import telebot
import sched, time
from generator import Generator

token = open('tgbot_token.txt').read()
chat_id = '@wandering_tokamak'
bot = telebot.TeleBot(token)

def send_to_channel(scheduler):
    print('started')
    while True: 
        try:
            gen = Generator()
            image = gen.generate()
            print('generated')
            bot.send_photo(chat_id=chat_id, photo=image)
            print('sent')
        except Exception as e:
            print(e)
    scheduler.enter(60, 1, send_to_channel, (scheduler,))


sch = sched.scheduler(time.time, time.sleep)
sch.enter(60, 1, send_to_channel, (sch,))
sch.run()