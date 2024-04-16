from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Загрузка токенизатора GPT-2 и предварительно обученной модели
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
# Установка токена для заполнения
tokenizer.pad_token = tokenizer.eos_token

# Путь к файлу JSON с датасетом
dataset_file = "DataSetRu.json"

# Загрузка датасета
dataset = load_dataset("json", data_files=dataset_file)["train"]

# Установка максимальной длины токенизации
max_length = 1024

# Токенизация и форматирование данных вручную
tokenized_dataset = []
for example in dataset:
    tokenized_example = tokenizer(
        example["question"],
        example["answer"],
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    tokenized_dataset.append(tokenized_example)

# Настройка аргументов для обучения
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Создание объекта Trainer и начало обучения
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=tokenized_dataset,
)
print("train starting")
trainer.train()

# Сохранение обученной модели
model.save_pretrained("your_trained_model")
