from llama_cpp import Llama

print("Loading model...")
llm = Llama(
    model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    verbose=False
)

print("✓ Model loaded successfully!")
print("\nTesting inference...")

response = llm(
    "Q: What is 5 + 7? A:",
    max_tokens=50,
    stop=["Q:", "\n"],
    temperature=0.7
)

print("Response:", response['choices'][0]['text'])
print("\n✓ Model is working!")
