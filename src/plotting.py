import matplotlib.pyplot as plt

class PlotTopicAndSubtopicDistributions():

    def __init__(self, news_dataframe):
        self.news_dataframe = news_dataframe

    def plot_topic_distrubtion(self):
        topics = self.news_dataframe['Topic'].value_counts().index[:10]
        distributions = self.news_dataframe['Topic'].value_counts()[:10]

        fig, ax = plt.subplots(figsize=(18,10))
        bar_values = ['index', 'values']
        ax.bar(topics, distributions, color='b')
        ax.set_title('Distribution of Topics')
        ax.set_ylabel('Number of news (log scaled)')
        plt.yscale('log');
        plt.savefig('topic_distribution.png')

        return None

    def plot_subtopic_distrubtion(self):
        subtopics = self.news_dataframe['Subtopic'].value_counts().index[:10]
        distributions = self.news_dataframe['Subtopic'].value_counts()[:10]

        fig, ax = plt.subplots(figsize=(18,10))
        bar_values = ['index', 'values']
        ax.bar(subtopics, distributions, color='orange')
        ax.set_title('Most Popular Subtopics')
        ax.set_ylabel('Number of news (log scaled)')
        plt.yscale('log');
        plt.savefig('subtopics_most_pop.png')

        return None
