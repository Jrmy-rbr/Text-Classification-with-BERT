import numpy as np
import pandas as pd

import sklearn as skl

import re
import string


stopwords = pd.read_csv('./stop-word-list.csv').columns.tolist()
stopwords = [word.split()[0] for word in stopwords if word.split()[0] != 'not' and word.split()[0] != 'no']

class CountMentionsInClass(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """Class defined to manage features related to mention in tweet: 
     which mentions are likely to be present into disaster tweets vs non disaster ones"""
    def __init__(self):
        self.all_mention_in_disaster = dict()
        self.all_mention_in_ndisaster = dict()
        self.shape_train_X = 0
        self.shape_train_y = 0
        
    def check_inputs(self,X, y=None, column=None):
        # checks the data and outputs the data such that type(X)=DataFrame and type(y)=Series
        
        X = pd.DataFrame(X) ; X = X.copy()
        if not y is None:
            y = pd.Series(y) ; y = y.copy()
            
        if type(column) not in [int, str, None]:
            raise TypeError("The parameter 'column' expects a string or an integer as input")
                     
        if column is None and X.shape[1] != 1:
                raise ValueError("You must specify the column's name")
        
        if y is None:
            return X
        else:
            return X, y
    
    
    def fit(self, X, y, column=None):
        """ Look in the train set which mentions are present in the desastrous tweet and in the 
        none-disastros tweets, and store these mentions in a dictionary for each category of tweet:
        - self.all_mention_in_disaster -> dict storing all mentions found in disastrous tweets
        - self.all_mention_in_ndisaster -> dict storing all metions found in none-disastrous tweets
        """
        X, y = self.check_inputs(X,y,column)
        self.shape_train_X = X.shape
        self.shape_train_y = y.shape
        
        if column is None:
            for idx, text in enumerate(X):
                mentions = re.findall("\s?@\w+", text)
                for mention in mentions:
                    if y.iloc[idx] == 1:
                        self.all_mention_in_disaster.update({mention:0})
                    elif y.iloc[idx] == 0:
                        self.all_mention_in_ndisaster.update({mention:0})
                    else:
                        raise ValueError("The target must be 0 or 1")
        else:
            if type(column) == int:
                column = X.columns[columun]
                
            for idx, text in enumerate(X[column]):
                mentions = re.findall("\s?@\w+", text) #find mentions
                for mention in mentions:
                    if y.iloc[idx] == 1:
                        self.all_mention_in_disaster.update({mention:0})
                    elif y.iloc[idx] == 0:
                        self.all_mention_in_ndisaster.update({mention:0})
                    else:
                        raise ValueError("The target must be 0 or 1")
        
        return self
        
    
    def transform(self, X, column=None):
        """ Looks for all the mentions in the input tweets of X, and counts the number of them 
        belonging to the dictionnaries self.all_mention_in_disaster & self.all_mention_in_ndisaster respectively.
        Then the method adds the number of mentions of the input tweet in self.all_mention_in_disaster 
        in a column and those in self.all_mention_in_ndisaster in another column. 
        """        
        X = self.check_inputs(X, column=column)
        if self.shape_train_X == 0:
            raise ValueError("The transformer has not been fitted yet")
        
        def count_mentions_in_disaster(text):
            mentions = re.findall("\s?@\w+", text)
            return sum(map(lambda mention: mention in self.all_mention_in_disaster, mentions))
        
        def count_mentions_in_ndisaster(text):
            mentions = re.findall("\s?@\w+", text)
            return sum(map(lambda mention: mention in self.all_mention_in_ndisaster, mentions))
        
        if type(column)==str:
            X['count_mentions_in_disaster'] = X[column].map(lambda text: count_mentions_in_disaster(text))
            X['count_mentions_in_ndisaster'] = X[column].map(lambda text: count_mentions_in_ndisaster(text))
        elif type(column)==int:
            X['count_mentions_in_disaster'] = X.iloc[:,column].map(lambda text: count_mentions_in_disaster(text))
            X['count_mentions_in_ndisaster'] = X.iloc[:,column].map(lambda text: count_mentions_in_ndisaster(text))
        else:
            X['count_mentions_in_disaster'] = X.apply(lambda text: count_mentions_in_disaster(text))
            X['count_mentions_in_ndisaster'] = X.apply(lambda text: count_mentions_in_ndisaster(text))
            
        return X
    
    
    def fit_transform(self,X, y, column =None):
        return self.fit(X, y, column=column).transform(X, column=column)
    
#----------------------------------------------------------------------------------
    
class CountTopNGramsInClass(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """Class defined to manage features related to the most present n-grams 
    in each class (disaster vs non-disaster) 
    """
    def __init__(self, n=1, n_ngrams=100):
        self.top_ngrams_in_disaster = dict()
        self.top_ngrams_in_ndisaster = dict()
        self.shape_train_X = 0
        self.shape_train_y = 0
        self.n = n
        self.n_ngrams = n_ngrams
        
    def check_inputs(self,X, y=None, column=None):
        # checks the data and outputs the data such that type(X)=DataFrame and type(y)=Series
        
        X = pd.DataFrame(X) ; X = X.copy()
        if not y is None:
            y = pd.Series(y) ; y = y.copy()
            
        if type(column) not in [int, str, None]:
            raise TypeError("The parameter 'column' expects a string or an integer as input")
                     
        if column is None and X.shape[1] != 1:
                raise ValueError("You must specify the column's name")
        
        if y is None:
            return X
        else:
            return X, y
    
    def make_ngrams_(self, text):
        token = [token for token in text.lower().split(' ') if token != '' if token not in stopwords]
        ngrams = zip(*[token[i:] for i in range(self.n)])
        return [' '.join(ngram) for ngram in ngrams]
        
        
    
    def fit(self, X, y, column=None):
        """ Look in the train set which ngram are present in the desastrous tweet and in the 
        none-disastros tweets, and store these ngrams in a dictionary for each category of tweet:
        - self.top_ngrams_in_disaster -> dict storing the top n_ngrams ngrams found in disastrous tweets
        - self.top_ngrams_in_ndisaster -> dict storing the top n_ngrams ngrams found in none-disastrous tweets
        """
        X, y = self.check_inputs(X,y,column)
        self.shape_train_X = X.shape
        self.shape_train_y = y.shape
        
        if column is None:
            for idx, text in enumerate(X):
                ngrams = self.make_ngrams_(text)
                for ngram in ngrams:
                    if y.iloc[idx] == 1:
                        if ngram in self.top_ngrams_in_disaster:
                            self.top_ngrams_in_disaster[ngram]+=1
                        else:
                            self.top_ngrams_in_disaster.update({ngram:1})
                    elif y.iloc[idx] == 0:
                        if ngram in self.top_ngrams_in_ndisaster:
                            self.top_ngrams_in_ndisaster[ngram]+=1
                        else:
                            self.top_ngrams_in_ndisaster.update({ngram:1})
                    else:
                        raise ValueError("The target must be 0 or 1")
        else:
            if type(column) == int:
                column = X.columns[columun]
                
            for idx, text in enumerate(X[column]):
                ngrams = self.make_ngrams_(text)
                for ngram in ngrams:
                    if y.iloc[idx] == 1:
                        if ngram in self.top_ngrams_in_disaster:
                            self.top_ngrams_in_disaster[ngram]+=1
                        else:
                            self.top_ngrams_in_disaster.update({ngram:1})
                    elif y.iloc[idx] == 0:
                        if ngram in self.top_ngrams_in_ndisaster:
                            self.top_ngrams_in_ndisaster[ngram]+=1
                        else:
                            self.top_ngrams_in_ndisaster.update({ngram:1})
                    else:
                        raise ValueError("The target must be 0 or 1")
                        
        # sort an keep only the top n_ngrams                
        self.top_ngrams_in_disaster = {key: value for idx, (key, value) in enumerate(sorted(self.top_ngrams_in_disaster.items(), key=lambda items: items[1], reverse=True)) if idx < self.n_ngrams}
        self.top_ngrams_in_ndisaster = {key: value for idx, (key, value) in enumerate(sorted(self.top_ngrams_in_ndisaster.items(), key=lambda items: items[1], reverse=True)) if idx < self.n_ngrams}
        
        return self
        
    
    def transform(self, X, column=None):
        """ Looks for all the ngrams in the input tweets of X, and counts the number of them 
        belonging to the dictionnaries self.top_ngrams_in_disaster & self.top_ngrams_in_ndisaster respectively.
        Then the method adds the number of ngrams of the input tweet in self.top_ngrams_in_disaster 
        in a column and those in self.top_ngrams_in_ndisaster in another column. 
        """    
        X = self.check_inputs(X, column=column)
        if self.shape_train_X == 0:
            raise ValueError("The transformer has not been fitted yet")
        
        def count_ngrams_in_disaster(text):
            ngrams = self.make_ngrams_(text)
            return sum(map(lambda ngram: ngram in self.top_ngrams_in_disaster, ngrams))
        
        def count_ngrams_in_ndisaster(text):
            ngrams = self.make_ngrams_(text)
            return sum(map(lambda ngram: ngram in self.top_ngrams_in_ndisaster, ngrams))
        
        if type(column)==str:
            X[f'count_{self.n}-grams_in_disaster'] = X[column].map(lambda text: count_ngrams_in_disaster(text))
            X[f'count_{self.n}-grams_in_ndisaster'] = X[column].map(lambda text: count_ngrams_in_ndisaster(text))
        elif type(column)==int:
            X[f'count_{self.n}-grams_in_disaster'] = X.iloc[:,column].map(lambda text: count_ngrams_in_disaster(text))
            X[f'count_{self.n}-grams_in_ndisaster'] = X.iloc[:,column].map(lambda text: count_ngrams_in_ndisaster(text))
        else:
            X[f'count_{self.n}-grams_in_disaster'] = X.apply(lambda text: count_ngrams_in_disaster(text))
            X[f'count_{self.n}-grams_in_ndisaster'] = X.apply(lambda text: count_ngrams_in_ndisaster(text))
            
        return X
    
    
    def fit_transform(self,X, y, column =None):
        return self.fit(X, y, column=column).transform(X, column=column)