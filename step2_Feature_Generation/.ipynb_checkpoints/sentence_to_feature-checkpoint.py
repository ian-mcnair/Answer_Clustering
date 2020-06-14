""" 
All of these functions work by passing in a single sentence and returning the created feature

Some features may require the entire dataset or additional information

To Do
[ ] Create a lemmatizer
[ ] Figure out if cosine similarity is possible

List of Working Features:
- Sentence Manipulation
    - Removing Stop words
    - Porter Stem
    - Alphabetical order
- Count
    - Word
    - Sentence
- Similarity Measures
    - Jaccard
    - Spacy
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

################### similarity measures ####################

def jaccard_similarity(student_answer, teacher_answer):
    a = set(student_answer.split(" "))
    b = set(teacher_answer.split(" "))
    c = a.intersection(b)
    return (len(c) / (len(a) + len(b) - len(c)))
    
def spacy_similarity(doc1, doc2):
    """
    Uses spacys built-in word embeddings to do similarity measures
    May need more testing as its probably not ideal if this is good at matching words
    and not phrases
    
    Is there a way I could match a phrase?
    """
    student_answer = nlp(doc1)
    teacher_answer = nlp(doc2)
    return student_answer.similarity(teacher_answer)

################### Entity Extraction ####################
def unigram_entity_extraction(df, sentence_col_name, answer):
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
    """
    This works, but doesn't havea high match rate.
    More matches with the normal sentence, but it doesn't
    correlate to prediction it seems
    """
    # Create list of bigrams for answer
    bigram_answer = create_list_of_bigrams(answer)
    
    # Need to compare bigrams to bigrams, below I am comparing
    # one word to bigram which is always going to return false.
    
    for bigram in bigram_answer:
        bigram_ = bigram.replace(" ", "_")
        df[f'has_{bigram_}'] = df[sentence_col_name].apply(lambda sent: int(bigram in sent))
        
    return df

def create_list_of_bigrams(sentence):
    sentence_list = sentence.split(" ")
    bigram_list = []

    #For each word in sentence, but needed the index
    for i in range(len(sentence_list)):
        # For index out of bounds error prrevention
        if i < len(sentence_list)-1:
            bigram_list.append(f"{sentence_list[i]} {sentence_list[i+1]}")
            
    return bigram_list

def trigram_entity_extraction(df, sentence_col_name, answer):
    trigram_answer = create_list_of_trigrams(answer)
   
    for trigram in trigram_answer:
        trigram_ = trigram.replace(" ", "_")
        df[f'has_{trigram_}'] = df[sentence_col_name].apply(lambda sent: int(trigram in sent))
        
    return df

def create_list_of_trigrams(sentence):
    sentence_list = sentence.split(" ")
    trigram_list = []

    #For each word in sentence, but needed the index
    for i in range(len(sentence_list)):
        # For index out of bounds error prrevention
        if i < len(sentence_list)-2:
            trigram_list.append(f"{sentence_list[i]} {sentence_list[i+1]} {sentence_list[i+2]}")
            
    return trigram_list

################### Feature Reduction ####################
def drop_column_if_all_same(df):
    """
    This scans each row and drops any column
    """
    df.loc[:, (df != 0).any(axis=0)]
    return df