import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ruta a tu modelo final de fase 1
MODEL_PATH = "./weights/final_rlm_lora"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def load_rlm_model():
    # TODO: Cargar el modelo base y el adaptador LoRA
    # print(f"Cargando modelo RLM desde {MODEL_PATH}...")
    # tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    # model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    # return model, tokenizer
    return None, None

def generate_reasoning(prompt, model, tokenizer):
    """
    Genera una respuesta que incluye el razonamiento (CoT).
    """
    # TODO: Implementar la generación
    # input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    # outputs = model.generate(**input_ids, max_new_tokens=512)
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return response
    return f"TODO: Implementar generación para el prompt: {prompt}"

if __name__ == "__main__":
    # Prueba local
    model, tokenizer = load_rlm_model()
    test_prompt = "Si tengo 3 manzanas y me dan el doble de las que tengo menos una, ¿cuántas tengo?"
    # print(generate_reasoning(test_prompt, model, tokenizer))
