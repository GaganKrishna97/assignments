import requests
import time
import psutil
from tqdm import tqdm

# Example endpoints (customize for your APIs)
HUGGINGFACE_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
OLLAMA_URL = "http://localhost:11434/api/generate"  # Replace with your Ollama REST endpoint

api_headers = {
    "Authorization": "Bearer YOUR_HF_API_KEY"  # Only for Hugging Face if needed
}

test_prompts = [
    "AI is transforming the world.",
    "This tool is not user friendly.",
    "I love exploring new technologies!",
    # ...add more prompts for robust benchmarking
]

def call_huggingface(text):
    response = requests.post(
        HUGGINGFACE_URL,
        headers=api_headers,
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    return response.json()

def call_ollama(text):
    response = requests.post(
        OLLAMA_URL,
        json={"prompt": text, "model": "llama2"}  # Update to your Ollama config
    )
    return response.json()

def measure_api(api_fn, prompts):
    latencies = []
    outputs = []
    for prompt in tqdm(prompts):
        start = time.time()
        result = api_fn(prompt)
        latency = time.time() - start
        latencies.append(latency)
        outputs.append(result)
    return latencies, outputs

def monitor_hardware(duration_sec=10):
    cpu_perc = psutil.cpu_percent(interval=None)
    mem_info = psutil.virtual_memory()
    print(f"CPU: {cpu_perc}% | RAM used: {mem_info.percent}%")

# ---- Benchmarking ----
print("Monitoring hardware (before):")
monitor_hardware()

# Hugging Face Benchmark
hf_latencies, hf_outputs = measure_api(call_huggingface, test_prompts)
print("Monitoring hardware (after Hugging Face):")
monitor_hardware()

# Ollama Benchmark
ol_latencies, ol_outputs = measure_api(call_ollama, test_prompts)
print("Monitoring hardware (after Ollama):")
monitor_hardware()

# ---- Metrics ----
def show_metrics(name, latencies, outputs):
    print(f"=== {name} ===")
    print(f"Mean Latency: {sum(latencies)/len(latencies):.3f}s")
    print(f"Min Latency: {min(latencies):.3f}s")
    print(f"Max Latency: {max(latencies):.3f}s")
    print(f"Throughput: {len(latencies) / sum(latencies):.2f} inferences/sec")
    # If you have ground truth, compute accuracy here!

show_metrics("Hugging Face", hf_latencies, hf_outputs)
show_metrics("Ollama", ol_latencies, ol_outputs)
