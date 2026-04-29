import telebot
import webbrowser


bot = telebot.TeleBot('8347000891:AAHpzj6oVWvwEJtMrsPyzwJt-z2KtDy2PUg')

@bot.message_handler(commands=['site'])
def site(message):
    webbrowser.open('https://sobaka.su/')

@bot.message_handler(commands=['vk'])
def site(message):
    webbrowser.open('http://vkontakte.ru/club4634260')


@bot.message_handler(commands=['phone'])
def main(message):
    bot.send_message(message.chat.id, '218-80-70')

@bot.message_handler(commands=['whatsapp'])
def main(message):
    bot.send_message(message.chat.id, '+79930045717')

@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, f'Здравстуйте, {message.from_user.first_name} {message.from_user.last_name}')


@bot.message_handler(commands=['help'])
def main(message):
    bot.send_message(message.chat.id, 'Комманды!')


@bot.message_handler()
def info(message):
    if message.text.lower() == 'привет':
        bot.send_message(message.chat.id, f'Здравстуйте, {message.from_user.first_name} {message.from_user.last_name}')
    elif message.text.lower() == 'id':
        bot.reply_to(message, f'ID: {message.from_user.id}')

@bot.message_handler(content_types=['photo'])
def get_photo(message):
    bot.reply_to(message, 'пам пам')

bot.polling(none_stop=True)

