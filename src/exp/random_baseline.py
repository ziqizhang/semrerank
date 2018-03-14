import json
import os
import re

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import graph.semrerank_doc_graph as tr
from util import utils
import graph.semrerank_doc_graph as td
import graph.semrerank_scorer as ts


def run_random_baseline(word2vec_model, jate_terms_file, stopwords, jate_terms_folder, out_folder):
    model=None
    if word2vec_model is not None:
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
        sum_unigram_scores[tu]=0.0
    sum_unigram_scores = utils.randomize(sum_unigram_scores)

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

embedding_model = None
jate_terms_file="/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/min1/cvalue.json"
stop = stopwords.words('english')
jate_terms_folder = "/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/min1_texpr"
out_folder="/home/zz/Work/data/texpr/texpr_output/acl_texpr_random"

# embedding_model = None
# jate_terms_file="/home/zz/Work/data/semrerank/jate_lrec2016/genia/min1/cvalue.json"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zz/Work/data/semrerank/jate_lrec2016/genia/min1_texpr"
# out_folder="/home/zz/Work/data/texpr/texpr_output/genia_texpr_random"

# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcw-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/min1/Basic.txt"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/vote"
# out_folder="/home/zqz/Work/data/semrerank/random/ttc_wind"

# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcm-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile/min1/cvalue.json"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile/vote"
# out_folder="/home/zqz/Work/data/semrerank/random/ttc_mobile"


# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia/min1/cvalue.json"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/genia/min1"
# out_folder="/home/zqz/Work/data/semrerank/random/genia"

# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia_atr4s/min1/Basic.txt"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/genia_atr4s/vote"
# out_folder="/home/zqz/Work/data/semrerank/random/genia"

# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_aclv2-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/min1/Basic.txt"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/vote"
# out_folder="/home/zqz/Work/data/semrerank/random/aclv2"

# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_aclv2-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/min1/Basic.txt"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/min1"
# out_folder="/home/zqz/Work/data/semrerank/random/aclv2"
run_random_baseline(embedding_model,jate_terms_file,stop, jate_terms_folder,out_folder)
