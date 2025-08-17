import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from textblob import TextBlob
import string

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create a sample corpus DataFrame
data = {
    'date': [
        '2023-01-10', '2023-02-15', '2023-03-20',
        '2023-04-25', '2023-05-30', '2023-06-15'
    ],
    'text': [
        'AI is transforming healthcare and outcomes are positive.',
        'Patients benefit from faster and more accurate diagnostics.',
        'Concerns about data privacy remain significant.',
        'New treatments using AI are saving lives.',
        'Acceptance of AI tools among doctors is growing.',
        'Some errors in AI predictions still cause issues.'
    ]
}
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# The rest of the script remains the same...
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

df['tokens'] = df['text'].astype(str).apply(preprocess)
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['month'] = df['date'].dt.to_period('M')
trend = df.groupby('month')['sentiment'].mean()

plt.figure(figsize=(8, 3))
trend.plot(marker='o')
plt.title('Average Sentiment Over Time')
plt.ylabel('Sentiment Polarity')
plt.xlabel('Month')
plt.grid()
plt.tight_layout()
plt.show()

all_tokens = [item for sublist in df['tokens'] for item in sublist]
freq_dist = Counter(all_tokens)
most_common = freq_dist.most_common(10)
print("Most Frequent Keywords/Topics:", most_common)

# Readability (requires textstat)
try:
    import textstat
    df['readability'] = df['text'].apply(lambda x: textstat.flesch_reading_ease(x))
    print("Average Flesch Reading Ease Score: %.2f" % df['readability'].mean())
    print("Sample scores:", df['readability'].head())
except ImportError:
    print("Install textstat via: pip install textstat")

plt.figure(figsize=(8,4))
words, counts = zip(*most_common)
plt.bar(words, counts)
plt.title("Top 10 Keywords")
plt.ylabel("Frequency")
plt.show()
