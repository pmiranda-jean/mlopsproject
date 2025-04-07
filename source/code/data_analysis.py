import matplotlib as plt
from wordcloud import WordCloud
from collections import Counter

def missing_value(data): 
   print("Missing Value Analysis")
   print(data.isna().sum())
   if data.isna().sum().all() == 0:
    print("There are no missing values")
   else:
    print("There are some missing values")
   return 0 

def value_count(data): 
   print("Value Counts")
   label_counts = data["label"].value_counts()
   print(label_counts)
   return 0 
   
def plot_label_distribution(labels, title="Label Distribution"):
    label_counts = labels.value_counts()

    plt.figure(figsize=(8, 5))
    label_counts.plot(kind='bar', color='lightcoral')
    plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=0)
    plt.show()

def generate_word_cloud(text, title="Word Cloud"):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

def plot_word_frequencies(text, top_n=20, title="Word Frequency Distribution"):
    words = ' '.join(text).split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(top_n)

    words, frequencies = zip(*common_words)

    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies, color='skyblue')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()