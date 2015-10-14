'''
    We loop over words in a dataset, and for each word, we look at a context window around the word.
    We generate pairs of (pivot_word, other_word_from_same_context) with label 1,
    and pairs of (pivot_word, random_word) with label 0 (skip-gram method).

    We use the layer WordContextProduct to learn embeddings for the word couples,
    and compute a proximity score between the embeddings (= p(context|word)),
    trained with our positive and negative labels.

    We then use the weights computed by WordContextProduct to encode words
    and demonstrate that the geometry of the embedding space
    captures certain useful semantic properties.

    Read more about skip-gram in this particularly gnomic paper by Mikolov et al.:
        http://arxiv.org/pdf/1301.3781v3.pdf

    Note: you should run this on GPU, otherwise training will be quite slow.
    On a EC2 GPU instance, expect 3 hours per 10e6 comments (~10e8 words) per epoch with dim_proj=256.
    Should be much faster on a modern GPU.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python skipgram_word_embeddings.py

    Dataset: 5,845,908 Hacker News comments.
    Obtain the dataset at:
        https://mega.co.nz/#F!YohlwD7R!wec0yNO86SeaNGIYQBOR0A
        (HNCommentsAll.1perline.json.bz2)
'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import theano
from six.moves import cPickle
import os, re, json

from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.embeddings import WordContextProduct, Embedding
from six.moves import range
from six.moves import zip

max_words = 50000  # vocabulary size: top 50,000 most common words in data
skip_top_words = 0  # ignore top 100 most common words
n_epochs = 1
n_dims = 100  # embedding space dimension

save = True
load_model = False
load_tokenizer = False
train_model = True

base_path = "~/machine_learning/Language-Translation/"
save_dir = os.path.expanduser(base_path + "models/")
data_path = os.path.expanduser(base_path + "data/en/") + "med.txt"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_load_fname = "en_med_skipgram_model.pkl"
model_save_fname = "en_med_skipgram_model.pkl"
tokenizer_fname = "en_med_tokenizer.pkl"


# text preprocessing utils
html_tags = re.compile(r'<.*?>')
to_replace = [('&#x27;', "'")]
hex_tags = re.compile(r'&.*?;')


def clean_comment(comment):
    c = str(comment.encode("utf-8"))
    c = html_tags.sub(' ', c)
    for tag, char in to_replace:
        c = c.replace(tag, char)
    c = hex_tags.sub(' ', c)
    return c


def text_generator(path=data_path):
    f = open(path)
    for i, l in enumerate(f):
        # comment_data = json.loads(l)
        # comment_text = comment_data["comment_text"]
        comment_text = l
        if i % 10000 == 0:
            print('Generated ' + str(i) + ' lines')
        yield comment_text
    f.close()


class SkipGramEmbedding(object):
    def __init__(self,
                 max_words=50000,
                 skip_top_words=0,
                 n_epochs=1,
                 n_dims=100,
                 window_size=4,
                 loss='mse',
                 optimizer='rmsprop'
                 ):
        """
        :param max_words: use onlu the n most common words in data
        :param skip_top_words: ignore top m ost common words
        :param n_epochs: number of training epochs
        :param n_dims: embedding space dimension
        :return: embedding model
        """
        self.max_words = max_words
        self.skip_top_words = skip_top_words
        self.n_epochs = n_epochs
        self.n_dims = n_dims

        self.tokenizer = text.Tokenizer(nb_words=self.max_words)
        self._is_tokenizer_fit = False
        self._word_index = None
        self._reverse_word_index = None

        self.window_size = window_size
        self.optimizer = optimizer
        self.loss = loss
        self.embedding_model = Sequential()
        self.embedding_model.add(
            WordContextProduct(self.max_words, proj_dim=self.n_dims, init="uniform"))
        self.embedding_model.compile(loss=loss, optimizer=optimizer)
        self._are_embeddings_fit = False

    def fit_tokenizer(self, text):
        print("Fitting tokenizer...")
        self.tokenizer.fit_on_texts(text)
        self._is_tokenizer_fit = True

        self._word_index = self.tokenizer.word_index
        self._reverse_word_index = dict([(v, k) for k, v in list(self._word_index.items())])

        return self

    def load_tokenizer(self, path):
        print('Load tokenizer...')
        self.tokenizer = cPickle.load(path, 'rb')
        self._is_tokenizer_fit = True
        self._word_index = self.tokenizer.word_index
        self._reverse_word_index = dict([(v, k) for k, v in list(self._word_index.items())])

    def save_tokenizer(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        cPickle.dump(self.tokenizer, path, "wb")

    def top_words(self, n):
        sorted_words = sorted(self.tokenizer.word_counts.iteritems(), key=lambda p: (-p[1], p[0]))
        return sorted_words[0:n]

    def load_embeddings(self, path):
        print('Loading embeddings...')
        self.embedding_model = cPickle.load(open(path, 'rb'))
        self._are_embeddings_fit = True
        return self

    def _fit_embeddings(self, text):
        sampling_table = sequence.make_sampling_table(max_words)

        for e in range(self.n_epochs):
            print('-' * 40)
            print('Epoch', e)
            print('-' * 40)

            progbar = generic_utils.Progbar(self.tokenizer.document_count)
            samples_seen = 0
            losses = []

            for i, seq in enumerate(self.tokenizer.texts_to_sequences_generator(text)):

                #MAKE SURE TOKENIZER AND FITTING ARE WORKING
                #if i < 5:
                #    print(map(lambda x: reverse_word_index[x], seq))

                # get skipgram couples for one text in the dataset
                couples, labels = sequence.skipgrams(seq, max_words,
                                                     window_size=self.window_size,
                                                     negative_samples=1.,
                                                     sampling_table=sampling_table)
                if couples:
                    # one gradient update per sentence (one sentence = a few 1000s of word couples)
                    X = np.array(couples, dtype="int32")
                    loss = self.embedding_model.train_on_batch(X, labels)
                    losses.append(loss)
                    if len(losses) % 100 == 0:
                        progbar.update(i, values=[("loss", np.mean(losses))])
                        losses = []
                    samples_seen += len(labels)
            print('Samples seen:', samples_seen)
        print("Training completed!")
        return self

    def save_embeddings(self,path):
        print("Saving model...")
        if not os.path.exists(path):
            os.makedirs(path)
        cPickle.dump(self.embedding_model, open(path, "wb"))


print("It's test time!")

# recover the embedding weights trained with skipgram:
weights = model.layers[0].get_weights()[0]

# we no longer need this
del model

weights[:skip_top_words] = np.zeros((skip_top_words, n_dims))
norm_weights = np_utils.normalize(weights)

word_index = tokenizer.word_index
reverse_word_index = dict([(v, k) for k, v in list(word_index.items())])


def embed_word(w):
    i = word_index.get(w)
    if (not i) or (i < skip_top_words) or (i >= max_words):
        return None
    return norm_weights[i]


def closest_to_point(point, nb_closest=3):
    proximities = np.dot(norm_weights, point)
    tups = list(zip(list(range(len(proximities))), proximities))
    tups.sort(key=lambda x: x[1], reverse=True)
    return [(reverse_word_index.get(t[0]), t[1]) for t in tups[:nb_closest]]


def closest_to_word(w, nb_closest=3):
    i = word_index.get(w)
    if (not i) or (i < skip_top_words) or (i >= max_words):
        return []
    return closest_to_point(norm_weights[i].T, nb_closest)


''' the resuls in comments below were for:
    5.8M HN comments
    dim_proj = 256
    nb_epoch = 2
    optimizer = rmsprop
    loss = mse
    max_features = 50000
    skip_top = 100
    negative_samples = 1.
    window_size = 4
    and frequency subsampling of factor 10e-5.
'''

words = [
    "article",  # post, story, hn, read, comments
    "3",  # 6, 4, 5, 2
    "two",  # three, few, several, each
    "great",  # love, nice, working, looking
    "data",  # information, memory, database
    "money",  # company, pay, customers, spend
    "years",  # ago, year, months, hours, week, days
    "android",  # ios, release, os, mobile, beta
    "javascript",  # js, css, compiler, library, jquery, ruby
    "look",  # looks, looking
    "business",  # industry, professional, customers
    "company",  # companies, startup, founders, startups
    "after",  # before, once, until
    "own",  # personal, our, having
    "us",  # united, country, american, tech, diversity, usa, china, sv
    "using",  # javascript, js, tools (lol)
    "here",  # hn, post, comments
]

for w in words:
    res = closest_to_word(w)
    print('====', w)
    for r in res:
        print(r)
