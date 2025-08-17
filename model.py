import pandas as pd
import pickle as pk
import re, string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from paths import DATASETS_DIR, OUTPUTS_DIR

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

class Model:
  def __init__(self):
    self.dataset = pd.read_csv(DATASETS_DIR + "/sample30.csv")
    self.lemmatizer = WordNetLemmatizer()
    
    self.count_vector = pk.load(open(OUTPUTS_DIR + "/count_vector.pkl", "rb"))
    self.tfidf_transformer = pk.load(open(OUTPUTS_DIR + "/tfidf_transformer.pkl", "rb"))

    self.model = pk.load(open(OUTPUTS_DIR + "/model.pkl", "rb"))
    self.recommendation = pk.load(open(OUTPUTS_DIR + "/recommendation.pkl", "rb"))

  def __clean_text(self, text):
    text = text.lower()
    
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if not any(char.isdigit() for char in word))
    text = re.sub(r' +', ' ',text)
    
    return text

  def __lemmatize_text(self, text):
    words = word_tokenize(text)
    lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

  def check_username(self, username):
    row_count = len(self.dataset[self.dataset["reviews_username"] == username])
    print("check_username :: row count = {}".format(row_count))
    if row_count == 0:
      return "{} not found in the dataset".format(username)
    else:
      return None

  def recommended_products(self, username, count):
    if username not in self.recommendation.index:
      return None

    product_list = self.recommendation.loc[username].sort_values(ascending=False)[0:count]
    print("recommended_products :: product_list = {}".format(product_list.index.tolist()))
    products = self.dataset[self.dataset.name.isin(product_list.index.tolist())]
    output = products[['name', 'reviews_text']]
    return output

  def top_products(self, products, count):
    products["reviews_lemmatized"] = products["reviews_text"].apply(self.__clean_text)
    products["reviews_lemmatized"] = products["reviews_lemmatized"].apply(self.__lemmatize_text)
    word_vector = self.count_vector.transform(products["reviews_lemmatized"])
    tfidf_vector = self.tfidf_transformer.transform(word_vector)
    products['predicted_sentiment'] = self.model.predict(tfidf_vector)
    print("top_products :: predicted_sentiment = {}".format(products['predicted_sentiment'].head()))
    
    total_product = products.groupby(['name']).agg('count')
    recommended_df = products.groupby(['name','predicted_sentiment']).agg('count')
    recommended_df = recommended_df.reset_index()
    merged_df = pd.merge(recommended_df, total_product['reviews_text'], on='name')
    merged_df['percentage'] = (merged_df['reviews_text_x']/merged_df['reviews_text_y'])*100
    merged_df = merged_df.sort_values(ascending=False, by='percentage')
    print("top_products :: final_df = {}".format(merged_df.head()))
    
    return list(merged_df[merged_df['predicted_sentiment'] == 1]['name'][:count])

model = Model()