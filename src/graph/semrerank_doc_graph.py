import json
import logging
import ntpath
import os
import queue
import re
import threading

import time

from pathlib import Path

import datetime

from nltk.corpus import stopwords

from exp import exp_loader_doc_graph
from exp import exp_loader_doc_graph as exp
from util import utils
import networkx as nx
import pickle as pk
from graph import semrerank_scorer as ts
import numpy as np

lock = threading.Lock()
logger = logging.getLogger(__name__)
logging.basicConfig(filename='semrerank.log', level=logging.INFO, filemode='w')


def jate_terms_iterator(jate_json_outfile):
    logger.info("Loading extracted terms by JATE...")
    json_data = open(jate_json_outfile).read()
    data = json.loads(json_data)
    count = 0
    if type(data) is dict:
        for k, v in data.items():
            yield k, v
    else:
        for term in data:
            count = count + 1
            try:
                yield term['string'], term['score']
            except TypeError:
                print("error")


def init_graph(graph_data_file):
    graph_data = utils.load_saved_model(graph_data_file)
    if (graph_data is not None):
        G = graph_data
    else:
        if not exp.DIRECTED_GRAPH:
            G = nx.Graph()
        else:
            G = nx.DiGraph()
    return (G, graph_data_file)


# return a map that contains 'jate term <---> normalized list of components found on the graph'
def graph_build_log(stats, term):
    logger.info("\t currently at term: {}, {} edges added for {} nodes...".
                format(term, stats[0], stats[1]))


def build_graph_from_terms(jate_output_from_a_file, jate_term_max_n,
                           embedding_model_keys, embedding_model,
                           topN, simT, G, global_similarity_lookup,stopwords, unigrams_from_all_terms):
    terms = utils.read_lines(jate_output_from_a_file)

    constraint_node_set=None
    if(exp_loader_doc_graph.RESTRICT_NODES_TO_DOCUMENT):
        constraint_node_set=terms_to_unigrams(terms, jate_term_max_n, stopwords)
        simT=determine_threshold(constraint_node_set, embedding_model, embedding_model_keys,100000)
    if(exp_loader_doc_graph.RESTRICT_NODES_TO_TERMS):
        constraint_node_set=unigrams_from_all_terms

    count = 0
    count_ngrams = 0
    count_ngrams_on_graph = 0

    processed_unigram_nodes = set()
    for term in terms:
        count = count + 1
        if (count % 5000 == 0):
            print("\t{}, {}".format(count, time.strftime("%H:%M:%S")))
        norm_parts = utils.normalize_string(term)

        count_ngrams += len(norm_parts)
        term_ngrams = utils.find_ngrams(norm_parts, jate_term_max_n)
        selected_parts = list()
        for term_ngram in term_ngrams:
            if term_ngram in processed_unigram_nodes:
                continue
            if exp_loader_doc_graph.REMOVE_STOPWORDS and term_ngram in stopwords:
                continue
            processed_unigram_nodes.add(term_ngram)
            # check if this part maps to a phrase that is present in the model
            norm_term_ngram = re.sub(r'[^a-zA-Z0-9,/\-\+\s_]', ' ',
                                     term_ngram).strip()  # pattern must keep '_' as word2vec model replaces space with _ in n gram
            if len(norm_term_ngram) > 1 and term_ngram in embedding_model_keys:
                selected_parts.append(term_ngram)
                # get similarity vector and add to graph
                stats = add_nodes_and_edges(G, term_ngram, embedding_model, topN, simT, constraint_node_set,
                                            global_similarity_lookup)
                if stats[0] > 0:
                    count_ngrams_on_graph += 1

        # if len(selected_parts) == 0:
        #     logger.info("{}: term has no componenets on graph".format(term[0]))
            # else:
            #     graph_build_log(stats,term[0])

    if exp_loader_doc_graph.ADD_NODES_RECURSIVE:
        add_nodes_and_edges_recursive(G, processed_unigram_nodes, embedding_model, topN, simT, 1,
                                      exp_loader_doc_graph.ADD_NODES_RECURSIVE_MAX_ITER)

    # print("unigrams={}, added to graph={}".format(count_ngrams, count_ngrams_on_graph))
    return [count_ngrams_on_graph, count_ngrams]


def terms_to_unigrams(terms, jate_term_max_n, stopwords):
    selected_words = list()
    for term in terms:
        norm_parts = utils.normalize_string(term)

        term_ngrams = utils.find_ngrams(norm_parts, jate_term_max_n)
        for term_ngram in term_ngrams:
            if exp_loader_doc_graph.REMOVE_STOPWORDS and term_ngram in stopwords:
                continue
            # check if this part maps to a phrase that is present in the model
            norm_term_ngram = re.sub(r'[^a-zA-Z0-9,/\-\+\s_]', ' ',
                                     term_ngram).strip()  # pattern must keep '_' as word2vec model replaces space with _ in n gram
            if len(norm_term_ngram) > 1:
                selected_words.append(term_ngram)

    # print("unigrams={}, added to graph={}".format(count_ngrams, count_ngrams_on_graph))
    return selected_words


def determine_threshold(unigrams, model, model_keys, max):
    return None


def add_nodes_and_edges(G, node, model, N, threshold, constrain_node_set, global_similarity_lookup):
    selected_similar=[]
    count_edge = 0
    count_node = 0

    if node in global_similarity_lookup.keys():
        selected_similar = global_similarity_lookup[node]
        for item in selected_similar:
            G.add_edge(node, item[0])
            count_edge = count_edge + 1
            count_node = count_node + 1
    else:
        similar = model.wv.most_similar(positive=node, topn=N)
        for item in similar:
            if item[1] < threshold:
                break
            if constrain_node_set is not None and item[0] not in constrain_node_set:
                continue

            if len(re.sub('[^0-9a-zA-Z]+', '', item[0])) > 1:
                selected_similar.append(item[0])
                G.add_edge(node, item[0])
                count_edge = count_edge + 1
                count_node = count_node + 1

        lock.acquire()
        try:
            global_similarity_lookup[node]=selected_similar
        finally:
            # Always called, even if exception is raised in try block
            lock.release()
    return (count_node, count_edge)


def add_nodes_and_edges_recursive(G, exclude_nodes, model, N, threshold, iter, max_iter):
    comps_prev = nx.number_connected_components(G)

    current_nodes = G.nodes()
    if len(current_nodes) == len(exclude_nodes):
        logger.info("RECURSIVE_GRAPH_BUILDER_STOP_BY:nodes converge, {}".format(iter))
        return False

    next_exclude_nodes = set()

    for node in current_nodes:
        if node not in exclude_nodes:
            next_exclude_nodes.add(node)
            similar = model.wv.most_similar(positive=node, topn=N)
            for item in similar:
                if item[1] >= threshold and len(re.sub('[^0-9a-zA-Z]+', '', item[0])) > 1:
                    G.add_edge(node, item[0])
                    # G.add_edge(item[0], node)
    next_exclude_nodes.update(exclude_nodes)

    iter += 1
    if iter == max_iter:
        logger.info("RECURSIVE_GRAPH_BUILDER_STOP_BY:max iter")
        return False

    if nx.is_connected(G):
        logger.info("RECURSIVE_GRAPH_BUILDER_STOP_BY:connected, {}".format(iter))
        return False
    comps_now = nx.number_connected_components(G)
    if comps_prev == comps_now:
        logger.info("RECURSIVE_GRAPH_BUILDER_STOP_BY:irreducible, {}".format(iter))
        return False

    return add_nodes_and_edges_recursive(G, next_exclude_nodes, model, N, threshold, iter, max_iter)


def init_personalized_vector(graph_nodes, sorted_jate_terms, topN, max_percentage):
    selected = 0
    initialized = set()
    init_vector = dict([(key, 0.0) for key in graph_nodes])

    max_init_vertices = max_percentage * len(graph_nodes)

    if (exp_loader_doc_graph.GS_TERMS_FILE== ""):
        gs_terms_list=[]
    else:
        gs_terms_list = utils.read_and_normalize_terms(exp_loader_doc_graph.GS_TERMS_FILE)
        #print("supervised graph")

    for key in sorted_jate_terms:
        selected = selected + 1

        key = exp_loader_doc_graph.lemmatizer.lemmatize(key).strip().lower()
        key = re.sub(r'[^a-zA-Z0-9,/\-\+\s]', ' ', key).strip()
        if len(gs_terms_list)>0 and len(key)>2:
            if key not in gs_terms_list:
                continue

        for unigram in utils.normalize_string(key):
            if len(re.sub('[^0-9a-zA-Z]+', '', unigram)) < 2:
                continue
            if unigram in graph_nodes:
                init_vector[unigram] = 1.0
                initialized.add(unigram)
        if selected >= topN:
            # logger.info("personalization init non zero elements:{}, from top {}, with max initialisable vertices {}"
            #       .format(initialized, selected, max_init_vertices))
            # print("personalization init non zero elements:{}, from top {}, with max initialisable vertices {}"
            #       .format(initialized, selected, max_init_vertices))
            break

    if len(initialized)==0:
        return [None,0]
    return [init_vector,len(initialized)]


def save_graph_data(graph, graph_out_file):
    path = Path(graph_out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    pk._dump(graph, open(graph_out_file, 'wb'))
    # pagerank_out_file=graph_out_file+".rank"
    # pk._dump(term_ranks, open(pagerank_out_file, 'wb'))


def read_ranks_from_cache(graph_data_file, personalization_seed):
    parent_folder = Path(graph_data_file).parent
    file = os.path.basename(graph_data_file)
    if personalization_seed is None:
        new_file_path = str(parent_folder) + "/ranks_pnl_/" + str(file)
    else:
        new_file_path = str(parent_folder) + "/ranks_pnl_" + str(personalization_seed) + "/" + str(file)
    return utils.load_saved_model(new_file_path)


def save_ranks_to_cache(graph_data_file, personalization_seed, ranks):
    parent_folder = Path(graph_data_file).parent
    file = os.path.basename(graph_data_file)
    if personalization_seed is None:
        new_file_path = str(parent_folder) + "/ranks_pnl_/" + str(file)
    else:
        new_file_path = str(parent_folder) + "/ranks_pnl_" + str(personalization_seed) + "/" + str(file)
    path = Path(new_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pk._dump(ranks, open(new_file_path, 'wb'))


#
# def calculate_oov(jate_terms_components, set_of_unigrams_as_nodes):
#     oov=[len(set_of_unigrams_as_nodes),0]
#     set_of_unigrams=set()
#     for key,value in jate_terms_components.items():
#         #covered=len(value)
#         unigrams=utils.normalize_string(key)
#         set_of_unigrams.update(unigrams)
#     oov[1]=len(set_of_unigrams)
#     return oov

def generate_term_component_map(jate_term_base_scores, jate_term_max_n, model):
    jate_terms_components = {}
    for term in jate_term_base_scores.keys():
        norm_parts = utils.normalize_string(term)
        term_ngrams = utils.find_ngrams(norm_parts, jate_term_max_n)
        selected_parts = list()
        for term_ngram in term_ngrams:
            # check if this part maps to a phrase that is present in the model
            norm_term_ngram = re.sub(r'[^a-zA-Z0-9,/\-\+\s_]', ' ',
                                     term_ngram).strip()  # pattern must keep '_' as word2vec model replaces space with _ in n gram
            if model is not None:
                if len(norm_term_ngram) > 1 and term_ngram in model:
                    selected_parts.append(term_ngram)
            else:
                if len(norm_term_ngram) > 1:
                    selected_parts.append(term_ngram)
        jate_terms_components[term] = selected_parts
    return jate_terms_components


def build_graph_batch(files, graph_data_folder,
                      jate_out_folder_per_file, model_keys, model, topN, simT,
                      personalized, sorted_seed_terms, stopwords, global_similarity_lookup, result):
    count = 0
    total_ngrams_added_to_graph = 0
    total_ngrams_from_terms = 0
    total_graph_connected = 0
    total_graph_components = 0
    total_files = 0
    total_nodes = 0
    total_edges = 0
    non_zero_elements_pnl_init=0
    sum_unigram_scores = {}

    unigrams_from_all_terms=set()
    if exp_loader_doc_graph.RESTRICT_NODES_TO_TERMS:
        for file in files:
            terms = utils.read_lines(jate_out_folder_per_file + "/" + file)
            unigrams_from_all_terms.update(terms_to_unigrams(terms, 5, stopwords))
        if exp_loader_doc_graph.RESTRICT_VOCAB_TO_TERMS:
            topN=int(len(unigrams_from_all_terms)*0.15)
        print("\tunigrams from all terms={}, revised topN={}".format(len(unigrams_from_all_terms), topN))

    for file in files:
        count += 1
        cached_graph_data = graph_data_folder + "/" + ntpath.basename(file)
        data = init_graph(cached_graph_data)
        if count % 100 == 0 or count == 1:
            print("\tThread={} processing {}\t{}, computing graph {}, time {}".format(
                threading.get_ident(), count, file, len(data[0]) == 0,
                str(datetime.datetime.now())))
        graph = data[0]
        save_graph = False
        if (len(graph) == 0):  # build the graph for this file for the first time
            ngram_coverage_stats =build_graph_from_terms(
                    jate_out_folder_per_file + "/" + file,5, model_keys, model, topN, simT,
                    graph, global_similarity_lookup, stopwords, unigrams_from_all_terms)
            total_ngrams_added_to_graph += ngram_coverage_stats[0]
            total_ngrams_from_terms += ngram_coverage_stats[1]

            if len(graph) != 0:
                if nx.is_connected(graph):
                    total_graph_connected += 1
                total_graph_components += nx.number_connected_components(graph)
            save_graph = True

        # update stats
        total_files += 1
        total_nodes += len(graph.nodes())
        total_edges += len(graph.edges())
        # jate_terms_components.update(jate_term_map_from_a_file)

        logger.info("{}: {} graph stats: nodes={}, edges={}".format(
            time.strftime("%H:%M:%S"), file, len(graph.nodes()), len(graph.edges())))

        semrerank = read_ranks_from_cache(cached_graph_data, personalized)
        if (semrerank is None):
            # personalization
            personalized_init = None
            if personalized is not None:
                output = \
                    init_personalized_vector(graph.nodes(), sorted_seed_terms, personalized,
                                             exp.MAX_PERCENTAGE_OF_VERTICES_FOR_PERSONALIZATION_INIT)
                personalized_init=output[0]
                non_zero_elements_pnl_init+=output[1]

            if exp_loader_doc_graph.LOG_PAGERAGE:
                logger.info("PageRank starts at {}".format(time.strftime("%H:%M:%S")))
            semrerank = nx.pagerank(graph,
                                    alpha=0.85, personalization=personalized_init,
                                    max_iter=5000, tol=1e-06)
            semrerank = utils.normalize(semrerank)
            if exp_loader_doc_graph.LOG_PAGERAGE:
                logger.info("\t completes at {}".format(time.strftime("%H:%M:%S")))
            save_ranks_to_cache(cached_graph_data, personalized, semrerank)

        # save
        if save_graph:
            save_graph_data(graph, data[1])

        # update scores for terms on the entire corpus
        for key, value in semrerank.items():
            existing_score = None
            if key in sum_unigram_scores.keys():
                existing_score = sum_unigram_scores[key]
            if existing_score is None:
                existing_score = 0.0
            existing_score += value
            sum_unigram_scores[key] = existing_score

    result.put([count, total_ngrams_added_to_graph, total_ngrams_from_terms,
                total_graph_connected, total_graph_components, total_files, total_nodes,
                total_edges, sum_unigram_scores, non_zero_elements_pnl_init])


def collect_and_log_results(results):
    count = 0
    total_ngrams_added_to_graph = 0
    total_ngrams_from_terms = 0
    total_graph_connected = 0
    total_graph_components = 0
    total_files = 0
    total_nodes = 0
    total_edges = 0
    sum_unigram_scores = {}
    non_zero_elements_pnl_init=0

    while not results.empty():
        res=results.get()
        count += res[0]
        total_ngrams_added_to_graph += res[1]
        total_ngrams_from_terms += res[2]
        total_graph_connected += res[3]
        total_graph_components += res[4]
        total_files += res[5]
        total_nodes += res[6]
        total_edges += res[7]
        non_zero_elements_pnl_init+=res[9]

        element_sum_unigram_scores = res[8]
        for key, value in element_sum_unigram_scores.items():
            if key in sum_unigram_scores.keys():
                v = sum_unigram_scores[key]
            else:
                v = 0
            v += value
            sum_unigram_scores[key] = v

    oov = [total_ngrams_added_to_graph, total_ngrams_from_terms]
    # finalize scores for terms on the entire corpus
    logger.info("COMPLETED ON ALL FILES, CALCULATING FINAL SCORES FOR ALL TERMS {}...".format(len(sum_unigram_scores)))
    logger.info(
        "Stats: total files={}, total nodes={}, total edges={}, oov={}/{}, graph_connected={}, graph_components={},"
        " personalization non-zero={}".format(
            count, total_nodes, total_edges, oov[0], oov[1], total_graph_connected, total_graph_components,
            non_zero_elements_pnl_init
        ))
    print(
        "Stats: total files={}, total nodes={}, total edges={}, oov={}/{}, graph_connected={}, graph_components={},"
        " personalization non-zero={}".format(
            count, total_nodes, total_edges, oov[0], oov[1], total_graph_connected, total_graph_components,
            non_zero_elements_pnl_init
        ))
    return sum_unigram_scores


def main(jate_json_outfile, jate_out_folder_per_file,
         model, topN, simT, jate_term_max_n, graph_data_folder, personalized, final_out_file,
         personalization_seed_file):
    model_keys = model.wv.vocab.keys()
    jate_term_base_scores = {c[0]: c[1] for c in jate_terms_iterator(jate_json_outfile)}
    jate_term_ttf={c[0]: c[1] for c in jate_terms_iterator(personalization_seed_file)}
    sorted_seed_terms = sorted(jate_term_ttf, key=jate_term_ttf.get, reverse=True)

    all_files = []
    for file in os.listdir(jate_out_folder_per_file):
        all_files.append(file)
    segs = np.array_split(all_files, exp_loader_doc_graph.CPU_CORES)
    result = queue.Queue()
    thread_list = []
    stop = stopwords.words('english')

    if exp_loader_doc_graph.TOPN_AUTO_DETERMINE:
        vocab_size=0
        for v in model_keys:
            norm = re.sub(r'[^a-zA-Z0-9,/\-\+\s_]', ' ',
                                  v).strip()
            if (norm in stop or len(norm) < 2):
                continue
            vocab_size+=1
        topN=int(round(vocab_size*0.15))
        #topN=3762
        logger.info("SELECTED_TOPN={}".format(topN))
        print("SELECTED_TOPN={}".format(topN))

    global_similarity_lookup={}
    for seg in segs:
        t = threading.Thread(target=build_graph_batch, args=(seg, graph_data_folder,
                                                             jate_out_folder_per_file, model_keys,
                                                             model, topN, simT, personalized, sorted_seed_terms,
                                                             stop,
                                                             global_similarity_lookup,
                                                             result))
        thread_list.append(t)

    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()

    sum_unigram_scores = collect_and_log_results(result)

    # calculate oov
    # oov=calculate_oov(jate_terms_components, set_of_unigrams_as_nodes)


    sum_unigram_scores = utils.normalize(sum_unigram_scores)
    #sum_unigram_scores=utils.randomize(sum_unigram_scores)
    jate_terms_components = generate_term_component_map(jate_term_base_scores,
                                                        jate_term_max_n,
                                                        model)

    term_rank_scores = ts.SemReRankScorer(sum_unigram_scores, jate_terms_components,
                                         jate_term_base_scores)

    # sorted_term_rank_scores = sorted(list(term_rank_scores), key=lambda k: k['score'])
    with open(final_out_file, 'w') as outfile:
        json.dump(list(term_rank_scores), outfile)
