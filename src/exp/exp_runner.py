import logging
import time

from gensim.models import Word2Vec
from graph import semrerank_alg as tr
from util import utils
from graph import semrerank_scorer as ts
import networkx as nx
import pickle as pk
import json
from exp import exp_loader

logger = logging.getLogger(__name__)
logging.basicConfig(filename='graph.log',level=logging.INFO, filemode='w')

def run(word2vec_model, jate_out_terms, topN_similar, similarity_threshold, jate_term_max_n,
        pagerank_out_file, final_out_file, personalized, exp_id):
    model = Word2Vec.load(word2vec_model)
    graph_data_file = pagerank_out_file + ".graph".format(str(similarity_threshold))
    setting_parts = graph_data_file.split("-")

    graph_data_file = setting_parts[0] + "-" + setting_parts[1] + "-" + setting_parts[2] + "-" + setting_parts[4]+"-" \
                      + setting_parts[5] + "-" + setting_parts[6] + "-" + setting_parts[7] + "-" + setting_parts[9]

    graph_data = utils.load_saved_model(graph_data_file)
    G = None
    jate_terms_components = None
    if (graph_data is not None):
        G = graph_data[0]
        jate_terms_components = graph_data[1]
    else:
        if not exp_loader.DIRECTED_GRAPH:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        jate_terms_components = \
            tr.build_graph(jate_out_terms, model, topN_similar, similarity_threshold, jate_term_max_n, G,
                           exp_loader.GRAPH_PRUNING)
        pk._dump((G, jate_terms_components), open(graph_data_file, 'wb'))

    jate_term_base_scores = {c[0]: c[1] for c in tr.jate_terms_iterator(jate_out_terms)}
    personalized_init = None
    if personalized is not None:
        personalized_init = tr.init_personalized_vector(G.nodes(), jate_term_base_scores, personalized)

    semrerank = utils.load_saved_model(pagerank_out_file)
    if semrerank is None:
        print("Graph size exp={}: {} {}".format(exp_id, len(G.nodes()), len(G.edges())))
        print(time.strftime("%H:%M:%S"))
        logger.info("Graph size exp={}: {} {}".format(exp_id, len(G.nodes()), len(G.edges())))
        logger.info(time.strftime("%H:%M:%S"))
        semrerank = nx.pagerank(G,
                                alpha=0.85, personalization=personalized_init,
                                max_iter=5000, tol=1e-06)
        print(time.strftime("%H:%M:%S"))
        print("Pagerank completes")
        logger.info(time.strftime("%H:%M:%S"))
        logger.info("Pagerank completes")
        semrerank = utils.normalize(semrerank)

    pk._dump(semrerank, open(pagerank_out_file, 'wb'))

    term_rank_scores = ts.SemReRankScorer(semrerank, jate_terms_components,
                                         jate_term_base_scores)

    # sorted_term_rank_scores = sorted(list(term_rank_scores), key=lambda k: k['score'])
    with open(final_out_file, 'w') as outfile:
        json.dump(list(term_rank_scores), outfile)


##### starting experiments
settings = exp_loader.create_settings()
for setting in settings:
    print("\nSETTING {}, {}".format(setting[8], setting[1]))
    run(setting[0], setting[1], setting[2], setting[3], setting[4],
        setting[5], setting[6], setting[7], setting[8])
