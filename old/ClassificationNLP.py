import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.metrics import accuracy_score
import sentence_to_feature as sf
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


class Classification_NLP:
    
    def __init__(self, data, doc, test_size = 'auto'):
        self.data = data
        self.doc = doc
        if test_size == 'auto':
            test_size = 1 - round(6/len(data),2)
            print(test_size)
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_split(test_size)
        self.teacher_answer = self.doc.teacher_answer.values[0]
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.doc['prediction'] = self.model.predict(self.data)
        self.score = -1
        self.new_answers = pd.DataFrame(columns = self.doc.columns.tolist() + self.data.columns.tolist())
        self.new_answers.drop(['label','question_id'], axis = 1, inplace=True)
        self.word_scaler = self.create_scaler(self.doc, 'student_answer', sf.word_count)
        self.sent_scaler = self.create_scaler(self.doc, 'student_answer', sf.sentence_count)
        
    
    def data_split(self, test_size):
         return train_test_split(self.data, self.doc.label, test_size = test_size, stratify = self.doc.label)
        
    def create_scaler(self, df, col, func):
        word_counts = df[col].apply(func)
        sc = MinMaxScaler()
        sc.fit_transform(word_counts.values.reshape(-1,1))
        return sc
        
    def accuracy(self, dataset = 'full'):
        test_set_score = accuracy_score(self.y_test, self.model.predict(self.X_test))
        self.score = accuracy_score(self.doc.label, self.doc.prediction)
        if dataset != 'full':
            return test_set_score
        else:
            return self.score
    
    def score_new_sentence(self, sentence_data):
        self.new_answers['prediction'] = self.model.predict(sentence_data)
        return self.new_answers['prediction'].values[-1]
    
    def create_features(self, answer):
        """
        NOTES
        The below actually has MORE features than what my documentation is showing
        Suggesting I left out features for some reason?
        
        Maybe when I was doing feature selection?
        
        Basically need to create all the features for the new answer
        """
        # Get the teacher's stuff
        a_answer = self.teacher_answer
        a_answer_ordered = sf.order_sentence(a_answer)
        a_stopwords = sf.remove_stopwords(self.teacher_answer)
        a_stemmed = sf.stem_sentence(a_stopwords)
        a_stopwords_ordered = sf.order_sentence(a_stemmed)
        a_stemmed_ordered = sf.order_sentence(a_stemmed)
        a_lem = sf.lemmatize_sentence(self.teacher_answer)
        a_lem_ordered = sf.order_sentence(a_lem)
        teacher_answers = [
#             a_answer,
#             a_answer_ordered,
#             a_stopwords,
#             a_stopwords_ordered,
            a_stemmed,
            a_stemmed_ordered,
#             a_lem,
#             a_lem_ordered,
        ]
        
        
        # Change sentence into multiple versions
        log = dict()
        log['student_answer'] = answer
        log['teacher_answer'] = self.teacher_answer
        log['q_answer'] = answer
        log['q_answer_ordered'] = sf.order_sentence(answer)
        log['q_stopwords'] = sf.remove_stopwords(answer)
        log['q_stopwords_ordered'] = sf.order_sentence(log['q_stopwords'])
        log['q_stemmed'] = sf.stem_sentence(answer)
        log['q_stem_ordered'] = sf.order_sentence(log['q_stemmed'])
        log['q_lemm'] = sf.lemmatize_sentence(answer)
        log['q_lemm_ordered'] = sf.order_sentence(log['q_lemm'])
        
        # Might need to save scaling until jsut before modeling
        log['wordcount'] = sf.word_count(answer)
        log['wordcount'] = sf.scale_column(self.word_scaler, log['wordcount'])
        log['sentence_count'] = sf.sentence_count(answer)
        log['sentence_count'] = sf.scale_column(self.sent_scaler, log['sentence_count'])
        #same fix as before

#         # # normal
#         log['normal_g_similarity'] = sf.generic_similarity(log['q_answer'], a_answer)
#         log['normal_j_similarity'] = sf.jaccard_similarity(log['q_answer'], a_answer)
#         log['normal_c_similarity'] = sf.cosine_similarity(log['q_answer'], a_answer)
#         # # Normal Ordered
#         log['normal_ordered_g_similarity'] = sf.generic_similarity(log['q_answer_ordered'], a_answer_ordered)
#         log['normal_ordered_j_similarity'] = sf.jaccard_similarity(log['q_answer_ordered'], a_answer_ordered)
#         log['normal_ordered_c_similarity'] = sf.cosine_similarity(log['q_answer_ordered'], a_answer_ordered)


#         Stem sim
        log['stem_g_similarity'] = sf.generic_similarity(log['q_stemmed'], a_stemmed)
        log['stem_j_similarity'] = sf.jaccard_similarity(log['q_stemmed'], a_stemmed)
        log['stem_c_similarity'] = sf.cosine_similarity(log['q_stemmed'], a_stemmed)
        # Ordered
        log['stem_ordered_g_similarity'] =  sf.generic_similarity(log['q_stem_ordered'], a_stemmed_ordered)
        log['stem_ordered_j_similarity'] =  sf.jaccard_similarity(log['q_stem_ordered'], a_stemmed_ordered)
        log['stem_ordered_c_similarity'] =  sf.cosine_similarity(log['q_stem_ordered'], a_stemmed_ordered)

#         Lem Sim
#         log['lem_g_similarity'] = sf.generic_similarity(log['q_lemm'], a_lem)
#         log['lem_j_similarity'] = sf.jaccard_similarity(log['q_lemm'], a_lem)
#         log['lem_c_similarity'] = sf.cosine_similarity(log['q_lemm'], a_lem)
# #        # Lem sim ordered
#         log['lem_ordered_g_similarity'] = sf.generic_similarity(log['q_lemm_ordered'], a_lem_ordered)
#         log['lem_ordered_j_similarity'] = sf.jaccard_similarity(log['q_lemm_ordered'], a_lem_ordered)
#         log['lem_ordered_c_similarity'] = sf.cosine_similarity(log['q_lemm_ordered'], a_lem_ordered)
        
        # Appending New Answer
        self.new_answers = self.new_answers.append(log, ignore_index = True)
        
        # Entity Extraction
        """
        I should use ALLL The combinations, ordered, stem/lem
        """
        types_of_sentences = [
#             'q_answer',
#             'q_answer_ordered',
#             'q_stopwords',
#             'q_stopwords_ordered',
            'q_stemmed',
            'q_stem_ordered',
#             'q_lemm',
#             'q_lemm_ordered',
        ]
        
        for sent_type, teach_ans in zip(types_of_sentences, teacher_answers):
            
            self.new_answers = sf.unigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
            self.new_answers = sf.bigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
            self.new_answers = sf.trigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
        

        
        