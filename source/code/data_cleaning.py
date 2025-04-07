import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

#Data Cleaning
def clean_data(data, text_column):
  stop_words = set(stopwords.words('english'))
  def clean_text(text):
    #The words removed from our datacleaning were added following the word cloud analysis presented later on. This is explained in the report.
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'film', '', text)
    text = re.sub(r'movie', '', text)
    text = re.sub(r'one', '', text)
    text = re.sub(r'show', '', text)
    text = re.sub(r'character', '', text)
    text = re.sub(r'br', '', text)
    text = re.sub(r'story', '', text)
    text = re.sub(r'see', '', text)
    text = re.sub(r'even', '', text)
    text = re.sub(r'make', '', text)
    text = re.sub(r'time', '', text)
    text = re.sub(r'scene', '', text)

    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

  data["cleaned_text"] = data[text_column].apply(clean_text)
  return data