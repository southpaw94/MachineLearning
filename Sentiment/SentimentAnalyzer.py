import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# PorterStemmer removes the suffixes of words, turning
# them all into the base form of the word
porter = PorterStemmer()

# stopwords includes a list of extremely common
# words in the English language which have negligible
# impact on the classification
stop = stopwords.words('english')

def tokenizer(text):
    return text.split()

# Returns the stems of words in a text string as a list
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def main():
    df = pd.read_csv('movie_data.csv')

    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values

    # term frequency --- inverse document frequency
    # assigns weights to each word based on the frequency
    # of each word in each string, while inversely weighing
    # terms according to frequency in entire document
    tfidf = TfidfVectorizer(strip_accents=None,
            lowercase=False,
            preprocessor=None)

    # Set up our parameters for the grid search, 48 various
    # parameter combinations in all
    param_grid = [{'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer,
            tokenizer_porter],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [1.0, 10.0, 100.0]},
        {'vect__ngram_range': [(1, 1)],
            'vect__stop_words': [stop, None],
            'vect__tokenizer': [tokenizer,
                tokenizer_porter],
            'vect__use_idf': [False],
            'vect__norm': [None],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [1.0, 10.0, 100.0]}
        ]

    lr_tfidf = Pipeline([('vect', tfidf),
        ('clf', LogisticRegression())])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
            scoring='accuracy',
            cv=5, verbose=1,
            n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)

    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
if __name__ == '__main__':
    main()
