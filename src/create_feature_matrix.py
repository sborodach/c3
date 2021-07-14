from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from time import time
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF, LatentDirichletAllocation

class CreateFeatureMatrix():
    '''Takes in language data and creates a feature matrix stored as an attribute of the name feature_matrix.
    Option to utilize the MIND dataset directly or other strings in list or Series form.

    Paramaters
    ----------
    features : str, ['LDA', 'NMF', 'TFIDF']
    n_components : int, must be at least 2
    ngram_range : tuple of two integers, first int must be equal to or less than the second

    Methods
    -------
    featurize : featurize corpus as TFIDF, LDA, or NMF vectors

    See Also
    --------

    Examples
    --------
    >>> data = ['This is a tool for building a content recommender',
                'Content recommenders can be challenging to evaluate',
                'Sports readers typically enjoy sports recommendations'
                'MIND is a userful dataset for studying recommenders',
                'medium.com is difficult to scrape from']
    >>> create_matrix = CreateFeatureMatrix(data, MIND=False, n_components=3)
    >>> create_matrix.featurize()
    >>> create_matrix.feature_matrix
        array([[0.70385031, 0.14807349, 0.1480762 ],
               [0.18583332, 0.64621002, 0.16795666],
               [0.33333333, 0.33333333, 0.33333333],
               [0.18583223, 0.16795668, 0.64621109],
               [0.33333333, 0.33333333, 0.33333333]])
    '''

    def __init__(self, data, MIND=True, ngram_range=(1,1), features='LDA', n_components=15):

        self.MIND = MIND
        if self.MIND:
            self.data = data
            self.corpus = data['content']
            self._add_stopwords()
        else:
            self.corpus = data

        self.vectorized = None
        self.ngram_range = ngram_range
        self.features = features
        self.stopwords = set(stopwords.words('english'))

        self.model = None
        self.feature_matrix = None

        self.n_components = n_components
        self.reconstruction_errors = {}
        self.feature_names = None


    def _add_stopwords(self):

        self.additional_stopwords = ['said', 'trump', 'just', 'like', '2019']
        for word in self.additional_stopwords:
            self.stopwords.add(word)

    def featurize(self):

        if self.features == 'LDA':
            self._LDA()

        elif self.features == 'NMF':
            self._NMF()

        elif self.features == 'TFIDF':
            self._vectorize()


    def _vectorize(self):
    #
        if self.features == 'LDA':
            tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                stop_words='english', ngram_range = self.ngram_range) # max_features=n_features
            self.vectorized = tf_vectorizer.fit_transform(self.corpus)
            self.feature_names = tf_vectorizer.get_feature_names()

        else:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=20,
                                               stop_words='english', ngram_range=self.ngram_range)
            self.vectorized = tfidf_vectorizer.fit_transform(self.corpus)
            self.feature_names = tfidf_vectorizer.get_feature_names()
            if self.features == 'TFIDF':
                self.feature_matrix = self.vectorized


    def _LDA(self):

        self._vectorize()

        self.model = LDA(n_components = self.n_components).fit(self.vectorized)
        self.feature_matrix = self.model.transform(self.vectorized)


    def _NMF(self):

        self._vectorize()

        self.model = NMF(n_components=self.n_components, max_iter=400, random_state=1,
              alpha=.1, l1_ratio=.5).fit(self.vectorized)

        self.feature_matrix = self.model.transform(self.vectorized)


    def _reconstruction_error(self): # get reconstruction error per n_components

        for n_components in range(5,30,5):

            if self.features == 'LDA':
                model = LDA(n_components=n_components).fit(self.vectorized)
                self.reconstruction_errors[n_components] = model.reconstruction_err_

            else:
                model = NMF(n_components=n_components, max_iter=400, random_state=1,
                  alpha=.1, l1_ratio=.5).fit(self.vectorized)
                self.reconstruction_errors[n_components] = model.reconstruction_err_


    def plot_reconstruction_error(self): #plots elbow plot to find optimal n_components

        self._reconstruction_error()

        plt.plot(self.reconstruction_errors.keys(), self.reconstruction_errors.values(), color='orange')
        plt.title('Reconstruction Error per Num Components')
        plt.xlabel('Num Components')
        plt.ylabel('Error')
        plt.xticks(ticks=[5,10,15,20,25], labels=[5,10,15,50,25]);

    def plot_top_words(self, n_top_words, title): # from sklearn documentation

        fig, axes = plt.subplots(3, 5, figsize=(30, 25), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(self.model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            if self.features == 'LDA':
                ax.barh(top_features, weights, height=0.7)
            else:
                ax.barh(top_features, weights, height=0.7, color='b')
            ax.set_title(f'Topic {topic_idx +1}',
                         fontdict={'fontsize': 30})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=20)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show();
        plt.savefig('nmf_15.png')
