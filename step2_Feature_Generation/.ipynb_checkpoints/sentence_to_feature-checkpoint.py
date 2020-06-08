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

# Sentence Manipulation #
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

def porter_stem(sentence):
    return " ".join([porter.stem(word) for word in sentence.split(" ")])

def stem_sentence(sentence, stem=True):
    # Strip punctuaton
    sentence = sentence.translate(sentence.maketrans("","", string.punctuation))
    #Remove Stop Words
    sentence = remove_stopwords(sentence)
    #Stem or Lem
    if stem:
        return porter_stem(sentence)
    else:
        return sentence
    
def order_sentence(sentence):
    sentence = sorted(sentence.split(" "))
    return " ".join(sentence)

# Generating general features #
def word_count(sentence):
    return len(sentence.split(" "))

def sentence_count(sentence):
    return len(sentence.split("."))
    
def jaccard_similarity(student_answer, teacher_answer):
    a = set(student_answer.split(" "))
    b = set(teacher_answer.split(" "))
    c = a.intersection(b)
    return (len(c) / (len(a) + len(b) - len(c)))
    
def spacy_similarity(doc1, doc2):
    student_answer = nlp(doc1)
    teacher_answer = nlp(doc2)
    return student_answer.similarity(teacher_answer)

# Entity Extraction #
def single_entity_extraction(df, sentence_col_name, answer):
    """
    This breaks the sentence using spaces
    and then creates features based one each word
    """
    # Break sentence into list
    answer_list = answer.split(" ")
    
    for word in answer_list:
        #Goes across each row
        df[f'has_{word}'] = df[sentence_col_name].apply(lambda sent: int(word in sent))
            
    return df

def bigram_entity_extraction(df, sentence_col_name, answer):
    # Create list of bigrams
    bigram_answer = create_list_of_bigrams(answer)
    
    # Need to compare bigrams to bigrams, below I am comparing
    # one word to bigram which is always going to return false.
    
    for bigram in bigram_answer:
        df[f'has_{bigram}'] = df[sentence_col_name].apply(lambda sent: int(bigram in sent))
        
    return df

def create_list_of_bigrams(sentence):
    sentence_list = sentence.split(" ")
    bigram_list = [f"*_{sentence_list[0]}"]
    
    for i in range(len(sentence_list)):
        
        if i < len(sentence_list)-1:
            print(i)
            bigram_list.append(f"{sentence_list[i]}_{sentence_list[i+1]}")
        else:
            print(i)
            bigram_list.append(f"{sentence_list[i]}_*")
            
    return bigram_list