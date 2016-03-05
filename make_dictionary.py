__author__ = 'Mark'
import re
import pickle


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


use_shitty_online_dict = False
if use_shitty_online_dict:
    original_lines = [line for line in open('data/english-spanish-2016-02-04.txt', 'r')]
    original_lines = [line for line in original_lines if line[0] != '#' and line[0] != '\n']

    p = re.compile('(;|[^\s]+|\t)')
    match_lists = [p.findall(line) for line in original_lines]


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

use_google_translate_en = True
if use_google_translate_en:
    original_lines = [line for line in open('data/in_vocab_en_words.txt', 'r')]
    original_lines = [line[0:-1] for line in original_lines if line[0] != '#' and line[0] != '\n']

    translated_lines = [line for line in open('data/in_vocab_en_words_translated.txt', 'r')]
    translated_lines = [line[:-1] for line in translated_lines if line[0] != '#' and line[0] != '\n']
    translated_lines.pop(14126)

    # Found error using
    #matched_indices = [i for i, (word1, word2) in enumerate(zip(original_lines,translated_lines)) if word1==word2]
    #for mi in matched_indices:
    #    print mi
    gt_en_2_es_dict = {word1: {word2} for word1, word2 in zip(original_lines, translated_lines)}
    pickle.dump(gt_en_2_es_dict, open('data/gt_en_2_es.pkl', 'w+'))

use_google_translate_es = True
if use_google_translate_es:
    original_lines = [line for line in open('data/in_vocab_es_words.txt', 'r')]
    original_lines = [line[0:-1] for line in original_lines if line[0] != '#' and line[0] != '\n']

    translated_lines = [line for line in open('data/in_vocab_es_words_translated.txt', 'r')]
    translated_lines = [line[:-1] for line in translated_lines if line[0] != '#' and line[0] != '\n']
    original_lines.pop(1725)
    original_lines.pop(11155)
    original_lines.pop(29134)


    # Found error using
    #with open('data/foo.txt', 'w+') as f:
    #   for pair in zip(original_lines, translated_lines):
    #        f.write(str(pair)+'\n')


    #matched_indices = [i for i, (word1, word2) in enumerate(zip(original_lines,translated_lines)) if word1==word2]
    #for mi in matched_indices:
    #    print mi

    print len(original_lines), len(translated_lines)

    gt_es_2_en_dict = {word1: {word2} for word1, word2 in zip(original_lines, translated_lines)}
    pickle.dump(gt_es_2_en_dict, open('data/gt_es_2_en.pkl', 'w+'))



