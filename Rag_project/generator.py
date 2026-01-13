# generator.py
from transformers import pipeline

def load_llm(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    return pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=300,
        temperature=0.2
    )

def generate_answer(llm, prompt: str):
    if not prompt.strip():
        raise ValueError("Prompt is empty")

    response = llm(prompt)[0]["generated_text"]
    return response.strip()
