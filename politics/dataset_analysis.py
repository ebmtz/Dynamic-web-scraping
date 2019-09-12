import pandas as pd
import pickle
import spacy
import gensim
import re
import pyLDAvis.gensim
import seaborn as sns
import matplotlib.pyplot as plt
from gensim import corpora
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from spacy.lang.en import English
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from politics.text_agent import TextAgent


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def tokenize(text):
    lda_tokens = []
    spacy.load('en')
    parser = English()
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def prepare_lda(text):
    stopset = set(stopwords.words('english'))
    stopset.update(['megathread'])
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in stopset]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def printif(log, condition):
    if condition: print(log)


def make_plots(data_df, filename):
    sns.set()
    plt.scatter(data_df['comp_sentiment'], data_df[['pos_sentiment']], label='Positive')
    plt.scatter(data_df['comp_sentiment'], data_df[['neg_sentiment']], label='Negative')
    plt.scatter(data_df['comp_sentiment'], data_df[['neu_sentiment']], label='Neutral')
    fig_title = '{} composite vs. components'.format(filename)
    plt.title(fig_title)
    plt.legend()
    plt.savefig('figs\\{} scatter.png'.format(fig_title))

    plt.clf()
    plt.subplot(1, 2, 1)
    components = ['pos_sentiment', 'neg_sentiment', 'neu_sentiment']
    for col in components:
        plt.hist(data_df[col], label=col, normed=True, alpha=0.5)
    plt.title('Sentiment Components Hist.')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(data_df['comp_sentiment'], label='comp_sentiment')
    plt.title('Composite Sentiment Hist.')
    plt.legend()

    plt.savefig('figs\\{} hist.png'.format(fig_title))

    plt.clf()


def execute(file, num_topics, trim_data=None, sim_threshold=.35, logging=True, LDA=True):
    filename = re.search(r'scrapes/(.*)_Results_.*\.csv', file).group(1)
    sid = SentimentIntensityAnalyzer()

    printif('Loading data...', logging)

    data_df = pd.read_csv(file, encoding="latin1", index_col=0).reset_index(drop=True)

    data_df['body'] = data_df['body'].str.replace('[^\x00-\x7F]', '')
    data_df['body'] = data_df['body'].replace(r'(\\n)', ' ', regex=True).replace(r' +', ' ', regex=True)

    if trim_data:
        printif('Trimming DataFrame...', logging)
        agent = TextAgent(trim_data, 'meeeh', sim_threshold)
        printif('\t- Similarity Threshold: {}'.format(agent.similarity_threshold), logging)
        printif('\t- Original DF size: {}'.format(data_df.shape[0]), logging)
        for index, row in data_df.iterrows():
            agent.should_append(pd.DataFrame({
                'title': [row['title']],
                'body': [row['body']],
            }), continual_save=False)
        data_df = agent.web_df
        printif('Trim Complete. New Size: {}'.format(data_df.shape[0]), logging)
    if data_df.empty:
        printif('No elements to analyze. Terminating.', logging)
        return

    data_df['pos_sentiment'] = 0.
    data_df['neg_sentiment'] = 0.
    data_df['neu_sentiment'] = 0.
    data_df['comp_sentiment'] = 0.
    data_df = data_df.reset_index(drop=True)
    printif('Analyzing sentiment...', logging)
    for index, row in data_df.iterrows():
        printif('\tAnalyzing... {}'.format(row['title']), logging)
        sent = sid.polarity_scores(row['body'])
        printif('\t\tSentiment: {}'.format(sent), logging)
        data_df.loc[index, 'pos_sentiment'] = sent['pos']
        data_df.loc[index, 'neg_sentiment'] = sent['neg']
        data_df.loc[index, 'neu_sentiment'] = sent['neu']
        data_df.loc[index, 'comp_sentiment'] = sent['compound']

    if sim_threshold == .35:
        printif('Plotting Sentiments...', logging)
        make_plots(data_df, filename)

    if LDA:
        printif('Preparing text for analysis...', logging)
        printif('\t- Sanitizing text data', logging)
        text_data = [prepare_lda(article) for article in data_df['body'].values]

        printif('\t- Building dictionary', logging)
        dictionary = corpora.Dictionary(text_data)

        printif('\t- Building dictionary', logging)
        corpus = [dictionary.doc2bow(text) for text in text_data]

        printif('Generating LDA Model...', logging)
        lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

        printif('Topics in model: ', logging)
        topics = lda.print_topics(num_words=4)
        for topic in topics:
            printif(topic, logging)

        printif('Preparing LDA Display...', logging)
        lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=True)
        similarity_stamp = '_SIM{}_'.format(sim_threshold*100) if trim_data else ''
        pyLDAvis.save_html(lda_display, 'analysis/{}_{}{}_lda.html'.format(data_df.shape[0],
                                                                           filename,
                                                                           similarity_stamp))

    printif('DONE', logging)


sims = [.35, .40, .50, .60]
for sim in sims:
    execute('scrapes/Fox_Muller_Results_04212019_2143.csv', 10, 'Muller.txt', sim, LDA=False)
    execute('scrapes/Fox_2020Elections_Results_04212019_2358.csv', 10, '2020Elections.txt', sim, LDA=False)

    execute('scrapes/Politico_MullerReport_Results_04212019_1806.csv', 10, 'Muller.txt', sim, LDA=False)
    execute('scrapes/Politico_2020Elections_Results_04212019_1836.csv', 10, '2020Elections.txt', sim, LDA=False)
