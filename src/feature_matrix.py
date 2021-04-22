import numpy as np
import pandas as pd
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('ggplot')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA


class CreateFeatureMatrix():

    def __init__(self, data, ngram_range=(1,1), features='LDA', n_components=15):
        
        self.data = data
        self.corpus = data['content']
        
        self.vectorized = None
        self.ngram_range = ngram_range
        self.features = features
        
        self.model = None
        self.feature_matrix = None
        
        self.n_components = n_components
        self.reconstruction_errors = {}
        self.feature_names = None

        
    def featurize(self):
        
        if self.features == 'LDA':
            self._LDA()
            
        elif self.features == 'NMF':
            self._NMF()
            
        else:
            self._vectorize()
    
    def _vectorize(self):

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

    
    def _NMF(self):
        
        self._vectorize()
        
        self.model = NMF(n_components=self.n_components, max_iter=400, random_state=1,
              alpha=.1, l1_ratio=.5).fit(self.vectorized)
        
        self.feature_matrix = self.model.transform(self.vectorized)
    
    
    def _LDA(self):

        self._vectorize()        
        
        self.model = LDA(n_components = self.n_components).fit(self.vectorized)
        self.feature_matrix = self.model.transform(self.vectorized)


    # research optimal number of components
    
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
            if self.LDA:
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