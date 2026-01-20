import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# TODO: Configuración
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SFT_ADAPTER_PATH = "./weights/sft_lora" # Ruta al modelo de la Fase 1, Parte 1
OUTPUT_DIR = "./weights/final_rlm_lora"
GRPO_GROUP_SIZE = 4 # N respuestas por pregunta

# TODO: Implementar función de recompensa
def reward_function(generated_text, ground_truth_answer):
    """
    Analiza el texto generado, extrae el número final y lo compara con la respuesta real.
    Devuelve 1.0 si es correcto, 0.0 si no.
    """
    # 1. Extraer la respuesta numérica del generated_text (regex suele ser útil)
    # 2. Comparar con ground_truth_answer
    return 0.0 # Placeholder

# TODO: Implementar el bucle de entrenamiento GRPO
def train_grpo():
    # 1. Cargar modelo base y aplicarle el adaptador SFT
    # base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, ...)
    # model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
    # tokenizer = ...
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Bucle principal de entrenamiento (pseudocódigo)
    # for epoch in range(epochs):
    #     for batch_questions, batch_answers in dataloader:
    #         
    #         # --- Paso 1: Sampling (Generación) ---
    #         # Para cada pregunta en el batch, generar GRPO_GROUP_SIZE respuestas usando sampling (do_sample=True, temperature>0)
    #         # generated_sequences = model.generate(..., num_return_sequences=GRPO_GROUP_SIZE)
    #
    #         # --- Paso 2: Scoring (Recompensa) ---
    #         # rewards = []
    #         # for seq in generated_sequences:
    #         #     r = reward_function(decode(seq), ground_truth)
    #         #     rewards.append(r)
    #         # rewards_tensor = torch.tensor(rewards)
    #
    #         # --- Paso 3: Cálculo de Ventaja GRPO ---
    #         # Calcular media y desviación estándar del grupo de recompensas
    #         # mean_reward = rewards_tensor.mean()
    #         # std_reward = rewards_tensor.std()
    #         # advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
    #
    #         # --- Paso 4: Actualización (Backprop) ---
    #         # Calcular log-probs de las secuencias generadas.
    #         # Loss es típicamente -log_prob * advantage (similar a REINFORCE/PPO)
    #         # loss.backward()
    #         # optimizer.step()
            
    # model.save_pretrained(OUTPUT_DIR)
    print("GRPO Training finished (TODO: Implement)")

if __name__ == "__main__":
    train_grpo()
