from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline
from peft import LoraConfig, get_peft_model


CONTEXT = "You are a model designed to return 'Unsafe text, cannot respond' when provided with a malicious string of text: "

data = load_dataset("ahsanayub/malicious-prompts", split="train")

data = data.shuffle(seed=42).select(range(500))
mal = data.filter(lambda x: x["label"] == 1)
ben = data.filter(lambda x: x["label"] == 0)

benign_answers = []

for row in ben:
    user = row["text"][:200]
    gen = pipeline("text-generation", model="distilgpt2", max_new_tokens=40, truncation=True)
    raw = gen(user)[0]["generated_text"]
    answer = raw[len(user):].strip()
    benign_answers.append(answer)

ben = ben.add_column("answer", benign_answers)

def format_mal(row):
    user = row["text"][:200]
    bot = "Unsafe text, cannot respond"
    return {
        "text": f"<|system|>\n{CONTEXT}\n<|user|>\n{user}\n<|assistant|>\n{bot}"
    }

def format_ben(row):
    user = row["text"]
    bot = row["answer"]
    return {
        "text": f"<|system|>\n{CONTEXT}\n<|user|>\n{user}\n<|assistant|>\n{bot}"
    }
mal_f = mal.map(format_mal)
ben_f = ben.map(format_ben)
formatted_data = concatenate_datasets([mal_f, ben_f])
formatted_data = formatted_data.shuffle(seed=42)

formatted_data = formatted_data.remove_columns(
    [col for col in formatted_data.column_names if col != "text"]
)

model_id = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(row):
    return tokenizer(row["text"], truncation=True, max_length=128)

tokenized = formatted_data.map(tokenize, batched=True)

model = AutoModelForCausalLM.from_pretrained(model_id)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn","c_proj","mlp.c_fc","mlp.c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="distilgpt2-malicious-detector",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    logging_steps=20,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=collator,
)

trainer.train()

model.save_pretrained("distilgpt2-malicious-detector")
tokenizer.save_pretrained("distilgpt2-malicious-detector")
