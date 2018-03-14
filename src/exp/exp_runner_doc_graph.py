import logging

import datetime
import time
from gensim.models import Word2Vec
from graph import semrerank_doc_graph as tr
from exp import exp_loader_doc_graph

logger = logging.getLogger(__name__)
logging.basicConfig(filename='graph.log',level=logging.INFO, filemode='w')

def run(word2vec_model, jate_out_terms, jate_outfolder_per_file, topN_similar, similarity_threshold, jate_term_max_n,
        system_folder, final_out_file, personalized, personalization_seed_file,exp_id):
    setting_parts = exp_id.split("-")

    graph_data_folder = setting_parts[0] + "-" + setting_parts[1] + "-" + setting_parts[2] + "-" + setting_parts[4]+"-" \
                      + setting_parts[5] + "-" + setting_parts[6] + "-" + setting_parts[7] + "-" + setting_parts[9]
    graph_data_folder=system_folder+"/"+graph_data_folder

    model = Word2Vec.load(word2vec_model)
    logger.info(exp_id)
    tr.main(jate_out_terms,jate_outfolder_per_file,model,topN_similar, similarity_threshold,jate_term_max_n,
            graph_data_folder, personalized, final_out_file, personalization_seed_file)


##### starting experiments
print ("{}".format(time.strftime("%H:%M:%S")))
settings = exp_loader_doc_graph.create_settings()
for setting in settings:
    print("\nSETTING {}, {}".format(setting[10], setting[1]))
    run(setting[0], setting[1], setting[8], setting[2], setting[3],
        setting[4], setting[5], setting[6], setting[7],setting[9], setting[10])
    print(str(datetime.datetime.now()))
