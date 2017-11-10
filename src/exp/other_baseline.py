import json
import os
import re

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import graph.semrerank_doc_base as tr
from util import utils
import graph.semrerank_doc_base as td
import graph.semrerank_scorer as ts


def run_textrank_baseline(word2vec_model, jate_terms_file, stopwords, jate_terms_folder, textrank_score_file, out_folder):
    textrankscores=load_textrank_scores(textrank_score_file)
    compute(textrankscores, word2vec_model, jate_terms_file, stopwords, jate_terms_folder, out_folder)


def run_kcrdc_baseline(word2vec_model, jate_terms_file, stopwords, jate_terms_folder, kcr_score_file, out_folder):
    kcr_lookup=load_kcr_scores(kcr_score_file)
    compute(kcr_lookup, word2vec_model, jate_terms_file,
            stopwords, jate_terms_folder, out_folder)


def load_textrank_scores(in_file):
    score_lookup={}
    with open(in_file, encoding="utf-8") as file:
        kcr_words = file.readlines()
        for item in kcr_words:
            splits = item.split(",")
            score_lookup[splits[0]]=float(splits[1])
    return score_lookup


def load_kcr_scores(in_file):
    score_lookup={}
    with open(in_file, encoding="utf-8") as file:
        kcr_words = file.readlines()
        for item in kcr_words:
            start=item.index('(')+1
            end=item.index('[')
            word = item[start:end].strip()

            score_start=item.rfind(',')+1
            score_end=len(item)-2
            score=float(item[score_start: score_end])
            score_lookup[word]=score
    return score_lookup


def compute(score_lookup, word2vec_model, jate_terms_file, stopwords, jate_terms_folder, out_folder):
    kcr_lookup=score_lookup

    model = Word2Vec.load(word2vec_model)
    jate_term_base_scores = {c[0]: c[1] for c in tr.jate_terms_iterator(jate_terms_file)}
    term_unigrams = set()
    for term in jate_term_base_scores.keys():
        norm_parts = utils.normalize_string(term)
        for part in norm_parts:
            part = re.sub(r'[^a-zA-Z0-9,/\-\+\s_]', ' ',
                          part).strip()
            if (part in stopwords or len(part) < 2):
                continue
            else:
                term_unigrams.add(part)

    sum_unigram_scores = {}
    for tu in term_unigrams:
        if tu in kcr_lookup.keys():
            sum_unigram_scores[tu]=kcr_lookup[tu]
        else:
            sum_unigram_scores[tu]=0.0

    sum_unigram_scores = utils.normalize(sum_unigram_scores)

    jate_terms_components = td.generate_term_component_map(jate_term_base_scores,
                                                        5,
                                                        model)

    for file in os.listdir(jate_terms_folder):
        jate_term_base_scores = {c[0]: c[1] for c in tr.jate_terms_iterator(
            jate_terms_folder+"/"+file)}
        term_rank_scores = ts.SemReRankScorer(sum_unigram_scores, jate_terms_components,
                                         jate_term_base_scores)
        out_file=out_folder+"/"+file+"-random"
        # sorted_term_rank_scores = sorted(list(term_rank_scores), key=lambda k: k['score'])
        with open(out_file, 'w') as outfile:
            json.dump(list(term_rank_scores), outfile)



#ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcw-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind"+ATE_ALG_SET+"/min1/Basic.txt"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# word_weight_file= "//home/zqz/Work/atr4s/experiments/output/ttcw_dc_word/PostRankDC.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/dc/ttcw"

# ATE_ALG_SET="" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcm-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile"+ATE_ALG_SET+"/min1/cvalue.json"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# word_weight_file= "//home/zqz/Work/atr4s/experiments/output/ttcm_dc_word/PostRankDC.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/dc/ttcm"

# ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_aclv2-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2"+ATE_ALG_SET+"/min1/Basic.txt"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# word_weight_file= "//home/zqz/Work/atr4s/experiments/output/acl_dc_word/PostRankDC.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/dc/aclrd_ver2"

ATE_ALG_SET="" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia"+ATE_ALG_SET+"/min1/cvalue.json"
jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/genia"+ATE_ALG_SET+"/min1"
stop = stopwords.words('english')
word_weight_file= "//home/zqz/Work/atr4s/experiments/output/genia_dc_word/PostRankDC.txt"
out_folder="/home/zqz/Work/data/semrerank/ate_output/dc/genia"
run_kcrdc_baseline(embedding_model, jate_terms_file, stop, jate_terms_folder, word_weight_file, out_folder)



#ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcm-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile"+ATE_ALG_SET+"/min1/Basic.txt"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# word_weight_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2/words_ttcm.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank/ttcm"

# ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcm-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind"+ATE_ALG_SET+"/min1/Basic.txt"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# word_weight_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2/words_ttcw.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank/ttcw"

# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_aclv2-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/min1/Basic.txt"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/min1"
# word_weight_file= "/home/zqz/Work/data/semrerank/textrank/v2/words_aclv2.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_textrank/aclrd_ver2"

# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia_atr4s/min1/Basic.txt"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/genia_atr4s/min1"
# word_weight_file= "/home/zqz/Work/data/semrerank/textrank/v2/words_genia.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_textrank/genia"



# run_textrank_baseline(embedding_model, jate_terms_file, stop, jate_terms_folder, word_weight_file, out_folder)
