import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка токенизатора и предварительно обученной модели GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("your_trained_model")  # Путь к вашей обученной модели

# Функция для генерации ответа на вопрос
def generate_response(question, max_length=50):
    input_text = "Question: " + question + " Answer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Генерация ответа с помощью модели GPT-2
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # Декодирование сгенерированного ответа
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Функция для команды /start
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я бот-консультант по техническим вопросам. Задавайте свои вопросы.')

# Функция для ответа на сообщения пользователя
def reply_message(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    response = generate_response(user_input)
    update.message.reply_text(response)

def main() -> None:
    # Инициализация телеграмм-бота
    updater = Updater("YOUR_BOT_TOKEN")  # Укажите свой токен бота

    # Получение диспетчера для регистрации обработчиков
    dp = updater.dispatcher

    # Регистрация обработчиков команд
    dp.add_handler(CommandHandler("start", start))

    # Регистрация обработчика для ответа на сообщения пользователя
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, reply_message))

    # Запуск бота
    updater.start_polling()

    # Остановка бота при нажатии Ctrl+C
    updater.idle()

if __name__ == '__main__':
    main()
