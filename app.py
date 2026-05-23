from transformers import pipeline

# Load the model from Hugging Face
print("Loading BERT financial sentiment model...")
classifier = pipeline(
    "text-classification",
    model="STUDENT-135080/NLP_"
)
print("Model loaded!\n")

# Label mapping
label_map = {
    "LABEL_0": "Negative 📉",
    "LABEL_1": "Neutral  ➡️",
    "LABEL_2": "Positive 📈"
}

def analyze(text):
    result = classifier(text)[0]
    label = label_map[result["label"]]
    confidence = round(result["score"] * 100, 2)
    print(f"Text      : {text}")
    print(f"Sentiment : {label}")
    print(f"Confidence: {confidence}%")
    print("-" * 50)

# Example usage
if __name__ == "__main__":
    examples = [
        "Apple stock surges 12% after record quarterly earnings beat",
        "Markets tumble as inflation data sparks recession fears",
        "Fed holds interest rates steady, signals cautious outlook",
        "Tesla reports massive losses, CEO under fire from investors",
        "Revenue increased by 20% year over year",
    ]

    print("=" * 50)
    print("  Financial Sentiment Analyzer — BERT")
    print("=" * 50 + "\n")

    for text in examples:
        analyze(text)

    # Interactive mode
    print("\nEnter your own text (or type 'quit' to exit):")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input:
            analyze(user_input)
