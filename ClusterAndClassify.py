import pandas as pd
import numpy as np
import sentence_to_feature as sf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
import streamlit as st

class Cluster_and_Classify:
    def __init__(self, df, doc, answer_row = -1):
        self.doc = doc.copy()
        self.df = df.copy()
        self.clf_df = df.copy()
        self.teacher_answer = self.doc.teacher_answer.values[0]
        self.answer_row = answer_row
        self.clf = LogisticRegression()
        self.closest = []
        self.furthest = []
        self.X = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = np.array([])
        self.X_test = pd.DataFrame()
        self.y_pred = []
        self.y_true = []
        self.word_scaler = self.create_scaler(self.doc, 'student_answer', sf.word_count)
        self.new_answers = pd.DataFrame(columns = self.doc.columns.tolist() + self.df.columns.tolist())
        self.new_answers.drop(['label','question_id'], axis = 1, inplace=True)

        
    def create_scaler(self, df, col, func):
        word_counts = df[col].apply(func)
        sc = MinMaxScaler()
        sc.fit_transform(word_counts.values.reshape(-1,1))
        return sc
    
    def find_train_set_idxs(self):
        i = 0
        self.df['distances'] = 0
        for value in self.df.iloc[self.answer_row, :].values[:-1]:
            self.df['distances'] = self.df['distances'] + ((self.df.iloc[:, i] - value) ** 2)
            i+=1
        self.df['distances'] = self.df['distances'] ** 0.5
        self.df.loc[int(self.answer_row),'distances'] = -1
        self.closest = self.df[(self.df.distances > 0)].nsmallest(3, 'distances').index.values.tolist()
        self.closest.append(self.answer_row)
        self.furthest = self.df[(self.df.distances > 0)].nlargest(3, 'distances').index.values.tolist()
    
            
    def create_train_test_sets(self):
        X_correct = self.clf_df[self.clf_df.index.isin(self.closest)].copy()
        X_correct['label'] = 1
        X_incorrect = self.clf_df[self.clf_df.index.isin(self.furthest)].copy()
        X_incorrect['label'] = 0
        self.X = pd.concat([X_correct, X_incorrect])
        self.X_train = self.X.iloc[:,:-1]
        self.y_train = self.X.iloc[:,-1]
        self.X_test = self.clf_df[~self.clf_df.index.isin(self.closest + self.furthest)].copy()
    
    def classify(self):
        self.clf = self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.clf_df)
        self.y_true = self.doc['label'].values
        return self.y_pred, self.y_true
    
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def f1_scorer(self):
        return f1_score(self.y_true, self.y_pred)
    
    def balanced_accuracy(self):
        return balanced_accuracy_score(self.y_true, self.y_pred, adjusted = True)
    
    def precision(self):
        return precision_score(self.y_true, self.y_pred)
    
    def recall(self):
        return recall_score(self.y_true, self.y_pred)
    
    def kappa(self):
        return cohen_kappa_score(self.y_true, self.y_pred)
    
    def confusion_mtx(self):
        return confusion_matrix(self.y_true, self.y_pred)
    
    def run(self):
#         self.cluster()
        self.find_train_set_idxs()
        self.create_train_test_sets()
        self.classify()
        
    def score_new_sentences(self, sentence_data):
        st.write(sentence_data)
        self.new_answers['prediction'] = self.clf.predict(sentence_data)    
        
    def create_features(self, answer):
        """
        NOTES
        The below actually has MORE features than what my documentation is showing
        Suggesting I left out features for some reason?
        
        Maybe when I was doing feature selection?
        
        Basically need to create all the features for the new answer
        """
        # Get the teacher's stuff
        a_stopwords = sf.remove_stopwords(self.teacher_answer)
        a_stemmed = sf.stem_sentence(a_stopwords)
        a_stemmed_ordered = sf.order_sentence(a_stemmed)
        teacher_answers = [
            a_stemmed,
            a_stemmed_ordered,
        ]
        
        # Change sentence into multiple versions
        log = dict()
        log['student_answer'] = answer
        log['teacher_answer'] = self.teacher_answer
        log['q_answer'] = answer
        log['q_stopwords'] = sf.remove_stopwords(answer)
        log['q_stemmed'] = sf.stem_sentence(answer)
        log['q_stem_ordered'] = sf.order_sentence(log['q_stemmed'])
        
        # Might need to save scaling until jsut before modeling
        log['wordcount'] = sf.word_count(answer)
        log['wordcount'] = sf.scale_column(self.word_scaler, log['wordcount'])


#         Stem sim
        log['stem_g_similarity'] = sf.generic_similarity(log['q_stemmed'], a_stemmed)
        log['stem_j_similarity'] = sf.jaccard_similarity(log['q_stemmed'], a_stemmed)
        log['stem_c_similarity'] = sf.cosine_similarity(log['q_stemmed'], a_stemmed)
        # Ordered
        log['stem_ordered_g_similarity'] =  sf.generic_similarity(log['q_stem_ordered'], a_stemmed_ordered)
        log['stem_ordered_j_similarity'] =  sf.jaccard_similarity(log['q_stem_ordered'], a_stemmed_ordered)
        log['stem_ordered_c_similarity'] =  sf.cosine_similarity(log['q_stem_ordered'], a_stemmed_ordered)


        
        # Appending New Answer
        self.new_answers = self.new_answers.append(log, ignore_index = True)
        
        # Entity Extraction
        types_of_sentences = [
            'q_stemmed',
            'q_stem_ordered',
        ]
        
        for sent_type, teach_ans in zip(types_of_sentences, teacher_answers):
            
            self.new_answers = sf.unigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
            self.new_answers = sf.bigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
            self.new_answers = sf.trigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
