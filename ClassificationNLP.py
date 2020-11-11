import pandas as pd
from sklearn.metrics import accuracy_score
import sentence_to_feature as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
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
        self.pred = self.model.predict(self.X_test)
        self.score = -1
        self.new_answers = pd.DataFrame(columns = self.doc.columns.tolist() + self.data.columns.tolist())
        self.sep = len(self.doc.columns.tolist()) - 2
        self.new_answers.drop(['label','question_id'], axis = 1, inplace=True)
        self.word_scaler = self.create_scaler(self.doc, 'student_answer', sf.word_count)
        self.sent_scaler = self.create_scaler(self.doc, 'student_answer', sf.sentence_count)
        
    
    def data_split(self, test_size):
        
        return train_test_split(self.data, self.doc.label, test_size = test_size, stratify = self.doc.label, random_state = 42)
        
    def create_scaler(self, df, col, func):
        word_counts = df[col].apply(func)
        sc = MinMaxScaler()
        sc.fit_transform(word_counts.values.reshape(-1,1))
        return sc
        
    def accuracy(self):
        # Doesn't work, says int not callable
        test_set_score = accuracy_score(self.y_test, self.model.predict(self.X_test))
        
        #Total score
        self.score = accuracy_score(self.doc.label, self.doc.prediction)
        return self.score, test_set_score
    
    def grade_new_answer(self):
        self.new_answers['prediction'] = self.model.predict(self.new_answers.iloc[:, self.sep:])
    
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
        """
        I should use ALLL The combinations, ordered, stem/lem
        """
        types_of_sentences = [
            'q_stemmed',
            'q_stem_ordered',
        ]
        
        for sent_type, teach_ans in zip(types_of_sentences, teacher_answers):
            
            self.new_answers = sf.unigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
            self.new_answers = sf.bigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
            self.new_answers = sf.trigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
        

        
        