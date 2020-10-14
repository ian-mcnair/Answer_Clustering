# Data
import pandas as pd
# Math Imports
import math
#Spacy Imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_lg")
#NLTK Imports
from nltk.stem.wordnet import WordNetLemmatizer
lm = WordNetLemmatizer()
from nltk.stem import PorterStemmer
porter = PorterStemmer()
#string imports
import string
from string import punctuation
import re
# Transformers and Feature Selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

class NLP_Feature_Creator:
    
    def __init__(self, df):
        self.df = df
        pass
    
    def clean_sentence(self, sentence):
        # Should adjust this for a dataframe
        sentence = sentence.replace(".","").lower().strip()
        sentence = sentence.translate(sentence.maketrans("","", string.punctuation))
        sentence = sentence.replace("  "," ")
        return sentence
    
    def remove_stopwords(self, doc):
        doc = clean_sentence(doc)
        my_doc = nlp(doc)
        token_list = []
        for token in my_doc:
            token_list.append(token.text)

        filtered_sentence =[] 
        for word in token_list:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word) 
        return " ".join(filtered_sentence)
    
    def generate_nlp_features(self):
        pass
    
    def save_features(self):
        pass