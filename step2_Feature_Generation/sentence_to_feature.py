""" 
All of these functions work by passing in a single sentence and returning the created feature

Some features may require the entire dataset or additional information

To Do
[ ] Create a lemmatizer
[ ] Figure out if cosine similarity is possible
"""
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_lg")

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


def create_word_count(sentence):
    return len(sentence.split(" "))
    
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


# from collections import Counter
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# def get_cosine_sim(*strs): 
#     vectors = [t for t in get_vectors(*strs)]
#     return cosine_similarity(vectors)
    
# def get_vectors(sentence):
#     sentece = sentence.replace(".", "")
#     text = sentence.split(" ")
#     vectorizer = CountVectorizer(text)
#     vectorizer.fit(text)
#     return vectorizer.transform(text).toarray()