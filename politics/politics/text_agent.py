import pandas as pd
import nltk
import numpy as np
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import re


class TextAgent:
    def __init__(self, text_file, scraper_name, sim_thresh=.35):
        with open(text_file, 'r') as file:
            data = file.read().replace('\n', '').lower()

        self.similarity_threshold = sim_thresh
        self.stopset = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words=self.stopset, ngram_range=(1, 5))
        self.text_model = self.vectorizer.fit_transform([data])
        self.web_df = pd.DataFrame(columns=['title', 'body'])
        filename = re.search(r'(.*)\.txt', text_file).group(1)
        self.log_filename = '{}_{}_Results_{}.csv'.format(scraper_name, filename,
                                                          datetime.now().strftime('%m%d%Y_%H%M'))

    def similarity(self, text_sample):
        compare = self.vectorizer.transform([text_sample])

        return cosine_similarity(compare, self.text_model)

    def should_append(self, test_df, continual_save=True):
        if self.similarity(test_df['body'].astype(str).values[0]) > self.similarity_threshold:
            self.web_df = pd.concat([self.web_df, test_df])
            if continual_save:
                with open(self.log_filename, 'w+') as f:
                    self.web_df.to_csv(f)
            return True

        return False

    def print_lsa(self):
        lsa = TruncatedSVD(n_components=5, n_iter=100)
        lsa.fit(self.text_model)
        terms = self.vectorizer.get_feature_names()

        for i, comp in enumerate(lsa.components_):
            termsInComp = zip(terms, comp)
            sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:15]
            print('Concept {}:'.format(i))
            for term in sortedTerms:
                print(term)
            print()
