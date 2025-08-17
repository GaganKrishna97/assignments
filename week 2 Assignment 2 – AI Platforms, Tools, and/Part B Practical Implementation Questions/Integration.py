from transformers import pipeline

# Load sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Example input data
texts = ["I love AI platforms!", "This tool is difficult to use."]

# Model inference
results = classifier(texts)

# Output
for text, result in zip(texts, results):
    print(f"Text: {text} - Sentiment: {result['label']}, Score: {result['score']:.2f}")
