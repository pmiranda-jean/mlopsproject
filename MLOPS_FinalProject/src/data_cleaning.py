import re #Needed to remove punctuation, URLs, mentions, digits, and hashtags

def clean_data(data, text_column):
    def clean_text(text):
        #Simple standardization of text
        text = text.lower()
        
        #To Remove punctuation, URLs, mentions, digits, and hashtags since comments from online 
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'\d+', '', text)
        
        #Strip any extra spaces
        text = ' '.join([word for word in text.split()])
        
        return text

    #Apply the cleaning function of the text column 
    data["cleaned_text"] = data[text_column].apply(clean_text)
    return data
