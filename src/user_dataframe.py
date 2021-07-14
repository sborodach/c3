import pandas as pd

class UserDataFrameProcessor():

    def __init__(self, behaviors_tsv_filepath):
        self.user_dataframe = (pd.read_csv(behaviors_tsv_filepath, sep='\t').T.
                          reset_index().T.reset_index(drop=True))

    def clean_user_dataframe(self):
        userDataFrameWrangler = UserDataFrameWrangler(self.user_dataframe)
        userDataFrameWrangler.clean_user_dataframe()
        self.user_dataframe = userDataFrameWrangler.user_dataframe

class UserDataFrameWrangler():

    def __init__(self, user_dataframe):
        self.user_dataframe = user_dataframe
        self.impression_news = {}
        self.target_news = {}
        self.impression = []

    def clean_user_dataframe(self):
        self.user_dataframe.drop([0], axis=1, inplace=True)
        self.user_dataframe.columns = ['Impression_ID', 'User_ID', 'Timestamp', 'History', 'Impressions']
        self.user_dataframe.drop(['Impression_ID', 'Timestamp', 'Impressions'], axis=1, inplace=True)
        self.user_dataframe.drop(self.user_dataframe[self.user_dataframe['History'].isna() == True].index, axis=0, inplace=True)
        self.user_dataframe['History'] = self.user_dataframe['History'].apply(lambda x: x.split(' '))
        impressions = self.user_dataframe['Impressions'].apply(lambda x: x.split(' '))
        for i, impression in enumerate(impressions):
            self.impression_news[impressions.index[i]] = []
            for article in impression:
                self.impression_news[impressions.index[i]].append(article[:-2])
                if '-1' in article:
                    self.target_news[impressions.index[i]] = article[:-2]

        self.user_dataframe['Impressions'] = self.impression_news.values()
        self.user_dataframe['Target'] = self.target_news.values()

if  __name__ == "__main__":
    userDataFrameProcessor = UserDataFrameProcessor("../data/behaviors_test.tsv")
    userDataFrameProcessor.clean_user_dataframe()
    print(userDataFrameProcessor.user_dataframe.head())
