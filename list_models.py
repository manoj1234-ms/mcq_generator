import os
import google.generativeai as genai

# Set your API key
os.environ["GEMINI_API_KEY"] = "AIzaSyAXgcXkqotvm7HbTXsPOMZwuB0-C3asW94"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# List available models
models = genai.list_models()
print("Available Gemini models:")
for model in models:
    print(f"- {model.name}: {model.description}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Supported methods: {model.supported_generation_methods}")
    print()
