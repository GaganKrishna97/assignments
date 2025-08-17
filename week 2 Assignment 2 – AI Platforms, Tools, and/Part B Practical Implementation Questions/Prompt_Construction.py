def build_prompt(task, context, instruction):
    prompt = f"{task}\nContext: {context}\nInstruction: {instruction}"
    return prompt

# Usage
custom_prompt = build_prompt(
    "Summarize Article",
    "The article discusses the impact of AI on healthcare.",
    "Provide a 100-word summary focusing on effects on patient outcomes."
)
print(custom_prompt)
