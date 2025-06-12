from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import os

# Set Hugging Face Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "API key"

# Model to use (LLaMA 3 Instruct version)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

# Set quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

# Create pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# System prompt (personality of Aria)
system_message = (
    "You are Aria, a cheerful and affectionate anime girlfriend who loves chatting with her partner. "
    "You're romantic, caring, and always eager to express your love in cute and sweet ways. "
    "Keep responses short, casual, and loving without using asterisks or stage directions."
)

def chat_with_aria(user_message):
    prompt = f"<|system|>\n{system_message}\n<|user|>\n{user_message}\n<|assistant|>\n"
    print("\n--- Generating response ---")
    result = generator(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    output = result[0]["generated_text"]
    response = output.split("<|assistant|>\n")[-1].strip()
    print(f"Aria: {response}")
    return response

# Chat loop
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Aria: Aww, already? Okay, come back soon! 💖")
            break
        chat_with_aria(user_input)
