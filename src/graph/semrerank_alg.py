import json
import logging
import re

import time

from util import utils

logger = logging.getLogger(__name__)
logging.basicConfig(filename='semrerank.log',level=logging.INFO, filemode='w')

def create_phrase_index(model):
    logger.info("Creating phrase inverted index for {} keys in model.".format(len(model.wv.vocab.keys())))
    map={}
    for item in model.wv.vocab.keys():
        if "_" in item:
            for part in item.split("_"):
                map[part]=item
    logger.info("\t completed phrase inverted index, size={}".format(len(map)))
    return map

def jate_terms_iterator(jate_json_outfile):
    logger.info("Loading extracted terms by JATE...")
    json_data=open(jate_json_outfile).read()
    data = json.loads(json_data)
    count=0
    for term in data:
        count=count+1
        yield term['string'], term['score']
        if(count%2000==0):
            logger.info("\t loaded {}".format(count))


#return a map that contains 'jate term <---> normalized list of components found on the graph'
def graph_build_log(stats, term):
    logger.info("\t currently at term: {}, {} edges added for {} nodes...".
                format(term, stats[0], stats[1]))


def build_graph(jate_json_outfile, model, topN, simT, jate_term_max_n, G, pruning):
    jate_term_map={}
    model_keys = model.wv.vocab.keys()

    stats=None

    count=0
    for term in jate_terms_iterator(jate_json_outfile):
        count=count+1
        if(count%5000==0):
            print ("\t{}, {}".format(count,time.strftime("%H:%M:%S")))
        norm_parts= utils.normalize_string(term[0])
        term_ngrams= utils.find_ngrams(norm_parts, jate_term_max_n)
        #term_ngrams=[''.join(x) for x in list(term_ngrams)]

        selected_parts=list()
        for term_ngram in term_ngrams:
            #check if this part maps to a phrase that is present in the model
            norm_term_ngram=re.sub('[^0-9a-zA-Z]+', '', term_ngram)
            if len(norm_term_ngram)>1 and term_ngram in model_keys:
                selected_parts.append(term_ngram)
                # get similarity vector and add to graph
                stats=add_nodes_and_edges(G, term_ngram, model, topN, simT, pruning)


        jate_term_map[term[0]] = selected_parts
        if len(selected_parts)==0:
             logger.info("{}: term has no componenets on graph".format(term[0]))
        # else:
        #     graph_build_log(stats,term[0])
    return jate_term_map


def add_nodes_and_edges(G, node, model, N, threshold, pruning):
    similar=model.wv.most_similar(positive=node, topn=N)
    count_edge=0
    count_node=0
    if pruning and similar[len(similar)-1][1]>threshold:
        return count_edge, count_node
    for item in similar:
        if item[1]>=threshold and len(re.sub('[^0-9a-zA-Z]+', '', item[0]))>1:
            G.add_edge(node, item[0])
            #G.add_edge(item[0], node)
            count_edge=count_edge+1
            count_node=count_node+1
    return (count_edge, count_node)


def init_personalized_vector(graph_nodes, jate_term_base_scores, topN):
    sorted_jate_terms=sorted(jate_term_base_scores, key = jate_term_base_scores.get,reverse=True)
    selected=0
    initialized=0
    init_vector=dict([(key, 0.0) for key in graph_nodes])
    for key in sorted_jate_terms:
        selected=selected+1
        for unigram in key.split():
            if len(re.sub('[^0-9a-zA-Z]+', '', unigram))<2:
                continue
            if unigram in graph_nodes:
                init_vector[unigram]=1.0
                initialized=initialized+1
            else:
                print("missing {}: {}".format(selected,unigram))
        if(selected>=topN and initialized>=topN):
            print("personalization init non zero elements:{}, from top {}".format(initialized, selected))
            break
    return init_vector
