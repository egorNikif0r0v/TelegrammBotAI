import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка токенизатора и предварительно обученной модели GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("your_trained_model")  # Путь к вашей обученной модели

# Функция для генерации ответа на вопрос
def generate_response(question, max_length=100):
    input_text = "Question: " + question + " Answer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Генерация ответа с помощью модели GPT-2
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # Декодирование сгенерированного ответа
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Цикл взаимодействия с моделью через консоль
while True:
    user_input = input("Вы: ")
    if user_input.lower() == "exit":
        break

    response = generate_response(user_input)
    print(f"Бот: {response}")
