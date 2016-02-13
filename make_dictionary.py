__author__ = 'Mark'
import re
import pickle

lines = [line for line in open('data/english-spanish-2016-02-04.txt', 'r')]
lines = [line for line in lines if line[0] != '#' and line[0] != '\n']

p = re.compile('(;|[^\s]+|\t)')
match_lists = [p.findall(line) for line in lines]


def process_matches(matches):
    mode = 0
    output = (set(), set(), set())
    word_delim = ';'
    cat_delim = '\t'
    prev = ""
    for match in matches:
        if match == cat_delim:
            output[mode].add(unicode(prev.lower(), encoding="utf-8"))
            prev = ""
            mode += 1
        elif match == word_delim or match == cat_delim:
            output[mode].add(unicode(prev.lower(), encoding="utf-8"))
            prev = ""
        else:
            if prev=="":
                prev = prev+match
            else:
                prev = prev+" "+match
    output[2].add(unicode(prev.lower(), encoding="utf-8"))
    return output

output_lists = [process_matches(matches) for matches in match_lists]
en_2_es = {}
es_2_en = {}

for en_set, es_set, cat_set in output_lists:
    for en_word in en_set:
        en_2_es[en_word] = (es_set, next(iter(cat_set)))
    for es_word in es_set:
        es_2_en[es_word] = (en_set, next(iter(cat_set)))

pickle.dump(en_2_es, open('data/en_2_es.pkl','w+'))
pickle.dump(es_2_en, open('data/es_2_en.pkl','w+'))