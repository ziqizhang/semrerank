import os
import re
import networkx as nx
from gensim.models import Word2Vec
from nltk.corpus import stopwords

import graph.semrerank_doc_graph as tr

from graph import semrerank_doc_graph
from graph.semrerank_alg import jate_terms_iterator
import pickle as pk

from util import utils


def count_mwt(in_file):
    with open(in_file) as f:
        lines = f.readlines()
        total=len(lines)
        mwt=0
        for l in lines:
            toks=l.split(" ")
            if len(toks)>1:
                mwt+=1

    print("total={}, mwt={}".format(total, mwt/total))


def analyse_coverage(semrerank_json_out, pageranks_file, jate_term_max_n):
    pageranks = pk.load(open(pageranks_file, "rb"))

    total_parts = 0
    total_parts_covered = 0
    total_terms_completely_covered = 0
    for term in jate_terms_iterator(semrerank_json_out):
        count = count + 1
        if (count % 1000 == 0):
            print(count)
        norm_parts = utils.normalize_string(term[0])
        # term_unigrams= term.split()
        # term_ngrams=[''.join(x) for x in list(term_ngrams)]

        length = 0
        covered_parts = 0
        for term_unigram in norm_parts:
            # check if this part maps to a phrase that is present in the model
            norm_term_ngram = re.sub('[^0-9a-zA-Z]+', '', term_unigram)

            if len(norm_term_ngram) > 1:
                total_parts += 1
                length += 1
                if term_unigram in pageranks.keys():
                    total_parts_covered += 1
                    covered_parts += 1
                    # get similarity vector and add to graph

        if length == covered_parts:
            total_terms_completely_covered += 1

    print("total parts in the candidiates={}, total covered={}, total candidates completely covered={}".
          format(total_parts, total_parts_covered, total_terms_completely_covered))


def analyse_graph_structure(folder, folder_base_pattern, out_file):
    simTs = set()
    topNs = set()
    for file in os.listdir(folder):
        graph_type_folder = os.path.basename(file)
        parts = graph_type_folder.split("-")
        topN = parts[len(parts) - 2]
        simT = parts[len(parts) - 1]
        simTs.add(simT)
        topNs.add(topN)

    simTs = sorted(simTs)
    topNs = sorted(topNs)

    matrix_connected_graph = []
    matrix_nodes = []
    matrix_edges = []

    count_row = 0
    for topN in topNs:
        print(topN)

        row_connected_graph = []
        row_nodes = []
        row_edges = []

        count = 0
        for simT in simTs:
            target_folder = folder + "/" + folder_base_pattern + "-" + topN + "-" + simT
            print(target_folder)
            graph_stats = calculate_grap_stats(target_folder)

            row_connected_graph.append(graph_stats[0])
            row_nodes.append(graph_stats[2])
            row_edges.append(graph_stats[3])

            count += 1

        matrix_connected_graph.append(row_connected_graph)
        matrix_nodes.append(row_nodes)
        matrix_edges.append(row_edges)

    header = "CONNECTIVITY," + ",".join(simTs) + "\n"
    f = open(out_file, 'w')
    f.write(header)
    write_matrix(f, matrix_connected_graph, topNs)
    f.write("\n\n")
    header = "NODES," + ",".join(simTs) + "\n"
    f.write(header)
    write_matrix(f, matrix_nodes, topNs)
    f.write("\n\n")
    header = "EDGES," + ",".join(simTs) + "\n"
    f.write(header)
    write_matrix(f, matrix_edges, topNs)
    f.write("\n\n")
    f.close()


def write_matrix(f, matrix, row_prefix):
    for i in range(0, len(row_prefix)):
        row_str = row_prefix[i]
        row = matrix[i]

        for c in range(0, len(row)):
            row_str += "," + str(row[c])
        row_str += "\n"
        f.write(row_str)


def calculate_grap_stats(target_folder):
    total_graph_connected = 0
    total_graph_components = 0
    total_graph_nodes = 0
    total_graph_edges = 0
    for file in os.listdir(target_folder):
        if os.path.isdir(file):
            continue

        target_file = target_folder + "/" + file
        if os.path.isdir(target_file):
            continue
        graph_data = semrerank_doc_graph.init_graph(target_folder + "/" + file)
        graph = graph_data[0]

        if len(graph) == 0:
            continue
        if nx.is_connected(graph):
            total_graph_connected += 1
        total_graph_components += nx.number_connected_components(graph)

        total_graph_nodes += len(graph.nodes())
        total_graph_edges += len(graph.edges()) * 2

    return [total_graph_connected, total_graph_components, total_graph_nodes, total_graph_edges]


def analyze_node_degree(folder, jate_terms_file, stopwords, out_folder, folder_base_pattern):
    simTs = set()
    topNs = set()
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

    for file in os.listdir(folder):
        graph_type_folder = os.path.basename(file)
        parts = graph_type_folder.split("-")
        topN = parts[len(parts) - 2]
        simT = parts[len(parts) - 1]
        simTs.add(simT)
        topNs.add(topN)

    simTs = sorted(simTs)
    topNs = sorted(topNs)

    for simT in simTs:
        print(simT)

        for topN in topNs:
            target_folder = folder + "/" + folder_base_pattern + "-" + topN + "-" + simT
            print(target_folder)
            node_stats = calculate_node_stats(target_folder, term_unigrams)

            anynode_degrees = node_stats[0]
            termnode_degrees = node_stats[1]

            out_file = out_folder + "/" + folder_base_pattern + "-" + topN + "-" + simT + ".csv"
            f = open(out_file, 'w')
            for i in range(0, len(anynode_degrees)):
                line = str(anynode_degrees[i])
                if len(termnode_degrees) > i:
                    line += "," + str(termnode_degrees[i])
                line += "\n"
                f.write(line)

            f.close()


def calculate_node_stats(target_folder, term_unigrams):
    anynode = []
    termnode = []
    count = 0
    for file in os.listdir(target_folder):
        count += 1
        if (count % 100 == 0):
            print(count)
        if os.path.isdir(file):
            continue

        target_file = target_folder + "/" + file
        if os.path.isdir(target_file):
            continue
        graph_data = semrerank_doc_graph.init_graph(target_folder + "/" + file)
        graph = graph_data[0]

        for node in graph.nodes():
            degree = graph.degree(node) * 2
            anynode.append(degree)

            if (node in term_unigrams):
                termnode.append(degree)

    return [anynode, termnode]


def analyse_threshold(word2vec_model, jate_terms_file, stopwords, out_file, term_only):
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

    f = open(out_file, 'w')
    print(len(term_unigrams))
    line="UNIGRAM, 0.9, 0.8, 0.7, 0.6, 0.5\n"
    f.write(line)
    simTs=[0.9, 0.8, 0.7, 0.6, 0.5]
    for unigram in term_unigrams:
        if unigram not in model.wv.vocab.keys():
            continue
        similar = model.wv.most_similar(positive=unigram, topn=100000)
        line="\""+unigram+"\","
        for simT in simTs:
            #print(simT)
            count=0
            for item in similar:
                if term_only and item[0] not in term_unigrams:
                    continue
                if item[1] < simT:
                    break
                count+=1
            line+=str(count)+","
        line=line+"\n"
        f.write(line)
    f.close()




# folder = "/home/zqz/Work/data/semrerank/graph/doc_based/genia-hyperparam"
# folder_base_pattern = "em_g-uni-sg-w3-m2,g_dn-pn"
# out_file = "/home/zqz/Work/data/semrerank/graph_stats.csv"
# analyse_graph_structure(folder, folder_base_pattern, out_file)

# folder="/home/zqz/Work/data/semrerank/graph/doc_based/genia-hyperparam"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia/min2/attf.json"
# out_folder="/home/zqz/Work/data/semrerank/graph_node_stats"
# folder_base_pattern="em_g-uni-sg-w3-m2,g_dn-pn"
# stop = stopwords.words('english')
# analyze_node_degree(folder, jate_terms_file, stop, out_folder, folder_base_pattern)

count_mwt("/home/zz/Work/data/jate_data/acl-rd-corpus-2.0/acl-rd-ver2-gs-terms.txt")
count_mwt("/home/zz/Work/data/jate_data/genia_gs/concept/genia_gs_terms_v2.txt")
count_mwt("/home/zz/Work/data/jate_data/ttc/gs-en-mobile-technology.txt")
count_mwt("/home/zz/Work/data/jate_data/ttc/gs-en-windenergy.txt")

exit(0)

embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcw-uni-sg-100-w3-m1.model"
jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind/ttf.json"
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_aclv2-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/ttf.json"
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia/ttf.json"
stop = stopwords.words('english')
out_file="/home/zqz/Work/data/semrerank/threshold-analysis.csv"
analyse_threshold(embedding_model,jate_terms_file,stop,out_file, True)
