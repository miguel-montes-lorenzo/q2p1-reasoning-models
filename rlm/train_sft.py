import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# TODO: Configuración del modelo y dataset
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # O un modelo más pequeño si es necesario
DATASET_NAME = "gsm8k"
OUTPUT_DIR = "./weights/sft_lora"

def formatting_prompts_func(example):
    output_texts = []
    # TODO: Implementar la lógica para formatear el dataset.
    # El objetivo es crear un string que contenga:
    # Prompt del usuario + Inicio de pensamiento (<think>) + Razonamiento + Fin de pensamiento (</think>) + Respuesta final.
    # Ejemplo conceptual:
    # text = f"User: {example['question']}\nAssistant: <think>{example['reasoning']}</think> La respuesta es {example['answer']}"
    # output_texts.append(text)
    return output_texts

def train():
    # 1. Cargar Modelo y Tokenizer (usar cuantización 4bit/8bit si es necesario)
    # tokenizer = ...
    # model = ...

    # 2. Configurar LoRA
    # peft_config = LoraConfig(...)

    # 3. Cargar Dataset
    # dataset = load_dataset(DATASET_NAME, split="train")

    # 4. Configurar Entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        # ... otros argumentos
    )

    # 5. Inicializar SFTTrainer
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset,
    #     peft_config=peft_config,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     formatting_func=formatting_prompts_func,
    # )

    # 6. Entrenar y guardar
    # trainer.train()
    # trainer.save_model(OUTPUT_DIR)
    print("SFT Training finished (TODO: Implement)")

if __name__ == "__main__":
    train()
