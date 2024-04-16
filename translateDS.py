
import jsonlines
from googletrans import Translator
import time

# Создание объекта переводчика
translator = Translator()

# Функция для перевода текста из английского на русский с обработкой ошибок
def translate_to_russian(text):
    while True:
        try:
            translation = translator.translate(text, src="en", dest="ru")
            if translation.text:
                return translation.text
            else:
                return "Translation failed"  # Возвращаем заглушку в случае ошибки перевода
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying translation in 5 seconds...")
            time.sleep(5)

# Пути к исходному и результирующему файлам
source_file = "small-117M.test.jsonl"  # Путь к вашему исходному JSONL файлу
result_file = "TestDataSetRu.jsonl"  # Путь к результирующему JSONL файлу

# Создание и открытие результирующего файла для записи
with jsonlines.open(result_file, "w") as writer:
    # Чтение исходного файла и запись переведенных данных в результирующий файл
    with jsonlines.open(source_file, "r") as reader:
        for line in reader:
            # Перевод текста на русский
            translated_text = translate_to_russian(line["text"])
            # Запись в результирующий файл
            writer.write({"id": line["id"], "text": translated_text, "length": line["length"], "ended": line["ended"]})
