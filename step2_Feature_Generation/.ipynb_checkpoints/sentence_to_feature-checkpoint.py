""" 
All of these functions work by passing in a single sentence and returning the created feature

Some features may require the entire dataset or additional information

To Do
[ ] Create a lemmatizer
[ ] Figure out if cosine similarity is possible
"""
#Spacy Imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_lg")

#NLTK Imports
from nltk.stem import PorterStemmer
porter = PorterStemmer()

#string imports
import string
from string import punctuation

def remove_stopwords(doc):
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


def word_count(sentence):
    return len(sentence.split(" "))

def sentence_count(sentence):
    return len(sentence.split(".")) + 1
    
def jaccard_similarity(student_answer, teacher_answer):
    a = set(student_answer.split(" "))
    b = set(teacher_answer.split(" "))
    c = a.intersection(b)
    return (len(c) / (len(a) + len(b) - len(c)))
    
def remove_stopwords(doc):
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

def spacy_similarity(doc1, doc2):
    student_answer = nlp(doc1)
    teacher_answer = nlp(doc2)
    return student_answer.similarity(teacher_answer)

def porter_stem(sentence):
    return " ".join([porter.stem(word) for word in sentence.split(" ")])

def reduce_sentence(sentence, stem=True):
    # Strip punctuaton
    sentence = sentence.translate(sentence.maketrans("","", string.punctuation))
    #Remove Stop Words
    sentence = remove_stopwords(sentence)
    #Stem or Lem
    if stem:
        return porter_stem(sentence)
    else:
        return sentence