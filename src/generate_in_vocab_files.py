from __future__ import print_function
__author__ = 'Mark'

from embedding import *

en_embedding = Embedding('data/polyglot-en.pkl')
es_embedding = Embedding('data/polyglot-es.pkl')

with open('data/in_vocab_en_words.txt', 'w+') as f:
    for word in en_embedding.words:
        print('{}\n'.format(word.encode('utf8')), file=f)

with open('data/in_vocab_es_words.txt', 'w+') as f:
    for word in es_embedding.words:
        print('{}\n'.format(word.encode('utf8')), file=f)

