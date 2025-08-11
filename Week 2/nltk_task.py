import nltk

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # <-- new in recent versions


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

text = """Machine learning and natural language processing are revolutionizing technology.
          They help in building intelligent systems capable of understanding human language."""

tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
pos_tags = pos_tag(filtered_tokens)

noun_count = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
verb_count = sum(1 for word, tag in pos_tags if tag.startswith('VB'))
adj_count = sum(1 for word, tag in pos_tags if tag.startswith('JJ'))

print("Original Tokens:", tokens)
print("Filtered Tokens:", filtered_tokens)
print("POS Tags:", pos_tags)
print(f"Noun count: {noun_count}, Verb count: {verb_count}, Adjective count: {adj_count}")
