class ContentRecommender():

    '''Generate recommendations for news news based on user input.
    Built on mind.py and feature_matrix.py. Can compare recommendations to
    target column to evaluate recommender performance.

    Paramaters
    -------------
    corpus : list of document
    data : DataFrame
    feature_matrix : TFIDF, LDA, or NMF matrix (num_documents x num_features)
    similarity_metric : str, default='cosine'
        'jaccard', 'pearson'

    Methods
    -------
    recommend :


    Example
    -------

    Notes
    -----

    See Also
    --------


    '''

    def __init__(self, news_dataframe, user_data, feature_matrix, similarity_metric='cosine'):

        self.news_dataframe = news_dataframe
        self.corpus = news_dataframe['content']
        self.user_dataframe = user_data
        self.feature_matrix = feature_matrix

        self.similarity_dict = {'cosine': cosine_similarity} # , 'jaccard': jaccard, 'pearson': pearson
        self.similarity_metric = similarity_metric

        self.feature_names = None


    def _calculate_covariance(self, user_vector=None):

        return pd.DataFrame(self.similarity_dict[self.similarity_metric](self.feature_matrix, user_vector), index = self.news_dataframe.index)


    def recommend(self, User_ID, by=None):

        news = ''.join(list(self.user_dataframe[self.user_dataframe['User ID'] == User_ID]['Read news'].values))

        news_ind = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in news)].index

        user_vector = self.feature_matrix.loc[news_ind].mean(axis=0).to_numpy().reshape(1,-1)

        of_interest_ind = self._calculate_covariance(self.feature_matrix.to_numpy(), user_vector).sort_values(by=0)[-10:].index[::-1]

        print(f'We recommend the following news: ')
        t = PrettyTable([' ', 'Title']) # create table of topic, interst level, and number recommended
        for i, title in enumerate(list(self.news_dataframe['title'][of_interest_ind].values)):
            t.add_row([i + 1, title])
        t.align = 'l'
        print(t)


    def evaluate(self):

        history_news = ''.join(list(self.user_dataframe[self.user_dataframe['User_ID'] == User_ID]['History'].values))
        history_ind = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in history_news)].index
        impression_news = ''.join(list(self.user_dataframe[self.user_dataframe['User_ID'] == User_ID]['Impressions'].values))
        impression_ind = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in impression_news)].index

        user_vector = self.feature_matrix.loc[history_ind].mean(axis=0).to_numpy().reshape(1,-1)
        impressions_vectors = self.feature_matrix.loc[impression_ind].mean(axis=0).to_numpy().reshape(1,-1)
        df = pd.DataFrame(self.similarity_dict[self.similarity_metric](impressions_vectors, user_vector)).sort_values(by=0)[-10:].index[::-1]

        return df, self.user_dataframe['Target'][self.user_dataframe['User_ID'] == User_ID],

    def recommend(self, User_ID, by=None):

        news = ''.join(list(self.user_dataframe[self.user_dataframe['User ID'] == User_ID]['Read news'].values))

        news_ind = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in news)].index

        user_vector = self.feature_matrix.loc[news_ind].mean(axis=0).to_numpy().reshape(1,-1)

        of_interest_ind = self._calculate_covariance(self.feature_matrix.to_numpy(), user_vector).sort_values(by=0)[-20:].index[::-1]

        return of_interest_ind

        R = self.news_dataframe['subtopic'][news_ind].values

        S = self.news_dataframe['title'][news_ind].values

        T = self.news_dataframe['subtopic'][of_interest_ind].values

        U = self.news_dataframe['title'][of_interest_ind].values

        return pd.DataFrame({'code': self.news_dataframe['code'][of_interest_ind].values, 'topic': self.news_dataframe['topic'][of_interest_ind].values, 'title': self.news_dataframe['title'][of_interest_ind].values}), self.news_dataframe[by][of_interest_ind].value_counts().index, self.news_dataframe[by][of_interest_ind].value_counts().values


    def evaulate_user(self, num_users, by='topic'):
        user_data = self.user_dataframe[self.user_dataframe['Read news'].apply(lambda x: len(x.split(' ')) > 100)]

        for i in range(num_users): # range(user_data.shape[0])
            user_read_news = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in user_data.iloc[i,1].split(' '))]
            d = {}
            for j, v in zip(user_read_news[by].value_counts().index, user_read_news[by].value_counts().values):
                d[j] = v

            w, x, y = self.recommend(user_data.iloc[i,0], by)

            d2 = {}
            for i, v in zip(x,y):
                d2[i] = v


            tot = sum(d.values())
            for k, v in d.items():
                d[k] = v/tot

#             good = False
#             for v in d.values():
#                 if v > .4:
#                     good == True

#             if good:
            d_topics = {}
            for k, v in d.items():
                if v > .4:
                    d_topics['H'] = [k]

                if v > .25 and v <= .4:
                    if 'M' not in d_topics:
                        d_topics['M'] = []
                    d_topics['M'].append(k)

                if v > .1 and v <= .25:
                    if 'L' not in d_topics:
                        d_topics['L'] = []
                    d_topics['L'].append(k)

                if v <= .1:
                    if 'VL' not in d_topics:
                        d_topics['VL'] = []
                    d_topics['VL'].append(k)

#             if 'H' in d_topics:
            t = PrettyTable([by, 'Interest', '%', '#']) # create table of topic, interst level, and number recommended
            z = 0
            for k, v in d_topics.items():
                for i in v:
                    if i in d2:
                        t.add_row([i.title(), k, round(list(d.values())[z], 2) * 100, d2[i]])
                    else:
                        t.add_row([i.title(), k, round(list(d.values())[z], 2) * 100, 0])
                    z+=1
            print(t)
            print('\n')

            print(w)
