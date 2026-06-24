# infer.py
# Load the fine-tuned LoRA adapter and test it interactively
# Run: python3 infer.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./lora-code-reviewer/checkpoint-250"

print("loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

device = "mps" if torch.backends.mps.is_available() else "cpu"
base_model = base_model.to(device)

print("loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("model ready\n")

def review_code(code: str) -> str:
    prompt = f"<|user|>\nreview this code: {code}</s>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )
        
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # extract just the assistant response
    if "<|assistant|>" in full:
        return full.split("<|assistant|>")[-1].strip()
    return full


if __name__ == "__main__":
    print("Code Reviewer — paste code and press Enter twice to submit. Type 'quit' to exit.\n")
    while True:
        print("Code to review:")
        lines = []
        while True:
            line = input()
            if line.lower() == "quit":
                exit()
            if line == "" and lines:
                break
            lines.append(line)
        code = "\n".join(lines)
        print("\nReview:")
        print(review_code(code))
        print()