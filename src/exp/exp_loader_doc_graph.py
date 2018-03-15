import numpy as np
import os
#process = psutil.Process(os.getpid())
from nltk import WordNetLemmatizer

#GS_TERMS_FILE="/home/zqz/Work/data/jate_data/ttc/gs-en-windenergy.txt"
#GS_TERMS_FILE="/home/zqz/Work/data/jate_data/ttc/gs-en-mobile-technology.txt"
GS_TERMS_FILE="/home/zz/Work/data/jate_data/acl-rd-corpus-2.0/acl-rd-ver2-gs-terms.txt"
#GS_TERMS_FILE="/home/zqz/Work/data/jate_data/genia_gs/concept/genia_gs_terms_v2.txt"


TOPN_AUTO_DETERMINE=True
RESTRICT_NODES_TO_DOCUMENT=False
RESTRICT_NODES_TO_TERMS=True
RESTRICT_VOCAB_TO_TERMS=True
CPU_CORES=1
REMOVE_STOPWORDS=True
ADD_NODES_RECURSIVE=False
ADD_NODES_RECURSIVE_MAX_ITER=2
SCORING_NORMALIZE_JATE_SCORE=True #normalise jate to 0~1
#SCORING_METHOD=2  #1=multiply base jate score; 2=add base jate score; 3=discard jate base score
DIRECTED_GRAPH = False
MAX_PERCENTAGE_OF_VERTICES_FOR_PERSONALIZATION_INIT = 0.5
LOG_PAGERAGE=False
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("init")#this forces lemmatizer to load, in order to avoid non-thread safe usage of lemmatizer


def create_setting(word2vec_model, jate_out_terms, topN_similar, similarity_threshold, jate_term_max_n,
                   pagerank_out_folder, final_out_file, topN_personalized, exp_id,
                   jate_outfolder_per_file, appended_list, personalization_seed_file):

    props = (word2vec_model, jate_out_terms, topN_similar, similarity_threshold, jate_term_max_n,
             pagerank_out_folder, final_out_file, topN_personalized, jate_outfolder_per_file,
             personalization_seed_file,exp_id)
    appended_list.append(props)


def create_settings():
    settings = list()
    root_folder = "/home/zz/Work/data/semrerank"

    # jate_terms_folder = "/home/zz/Work/data/semrerank/jate_lrec2016/genia/min1"
    # jate_outfolder_per_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia/min1_per_file"
    # system_folder="/home/zz/Work/data/semrerank/graph/doc_based/genia"
    # personalization_seed="/home/zz/Work/data/semrerank/jate_lrec2016/genia/ttf.json"
    # output_folder="output_genia"
    # embedding_setting="em_g-uni-sg-100-w3-m1"

    # jate_terms_folder = "/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/min1"
    # jate_outfolder_per_file = "/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/min1_per_file"
    # system_folder="/home/zz/Work/data/semrerank/graph/doc_based/aclrd_ver2_atr4s"
    # personalization_seed="/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/ttf.json"
    # output_folder="output_aclv2_atr4s"
    # embedding_setting="em_aclv2-uni-sg-100-w3-m1"

    # jate_terms_folder = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile/min1"
    # jate_outfolder_per_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile/min1_per_file"
    # system_folder="/home/zz/Work/data/semrerank/graph/doc_based/ttc_mobile"
    # personalization_seed="/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile/ttf.json"
    # output_folder="output_ttcm"
    # embedding_setting="em_ttcm-uni-sg-100-w3-m1"

    jate_terms_folder = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/min1"
    jate_outfolder_per_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/min1_per_file"
    system_folder="/home/zz/Work/data/semrerank/graph/doc_based/ttc_wind_atr4s"
    personalization_seed="/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/ttf.json"
    output_folder="output_ttcw_atr4s"
    embedding_setting="em_ttcw-uni-sg-100-w3-m1"


    for file in os.listdir(jate_terms_folder):
        for k in np.arange(0.5, 0.4, -0.2):
            topN = 1
            base = os.path.basename(file)
            ate = os.path.splitext(base)[0]

            # k = round(k, 1)
            # graph_setting = "g_dn-pn-top{}-pnl0-t".format(topN)
            # create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
            #                jate_terms_folder + "/" + str(file),
            #                topN,
            #                k,
            #                5,
            #                system_folder,
            #                "{}/{}/{}-{},{}{}.json".format(root_folder,output_folder,
            #                                                   ate, embedding_setting,
            #                                                   graph_setting, k),
            #                None,
            #                "{},{}{}".format(embedding_setting, graph_setting, k),
            #                jate_outfolder_per_file,
            #                settings, personalization_seed)

            #print(process.memory_info().rss)

            # pnl=50
            # graph_setting = "g_dn-pn-top{}-pnl50-t".format(topN)
            # create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
            #                jate_terms_folder + "/" + str(file),
            #                topN,
            #                k,
            #                5,
            #                system_folder,
            #                "{}/{}/{}-{},{}{}.json".format(root_folder,output_folder,
            #                                                   ate, embedding_setting,
            #                                                   graph_setting, k),
            #                pnl,
            #                "{},{}{}".format(embedding_setting, graph_setting, k),
            #                jate_outfolder_per_file,
            #                settings, personalization_seed)


            pnl=100
            graph_setting = "g_dn-pn-top{}-pnl100-t".format(topN)
            create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
                           jate_terms_folder + "/" + str(file),
                           topN,
                           k,
                           5,
                           system_folder,
                           "{}/{}/{}-{},{}{}.json".format(root_folder,output_folder,
                                                              ate, embedding_setting,
                                                              graph_setting, k),
                           pnl,
                           "{},{}{}".format(embedding_setting, graph_setting, k),
                           jate_outfolder_per_file,
                           settings, personalization_seed)

            pnl=200
            graph_setting = "g_dn-pn-top{}-pnl200-t".format(topN)
            create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
                           jate_terms_folder + "/" + str(file),
                           topN,
                           k,
                           5,
                           system_folder,
                           "{}/{}/{}-{},{}{}.json".format(root_folder,output_folder,
                                                              ate, embedding_setting,
                                                              graph_setting, k),
                           pnl,
                           "{},{}{}".format(embedding_setting, graph_setting, k),
                           jate_outfolder_per_file,
                           settings, personalization_seed)

            # pnl=500
            # graph_setting = "g_dn-pn-top{}-pnl500-t".format(topN)
            # create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
            #                jate_terms_folder + "/" + str(file),
            #                topN,
            #                k,
            #                5,
            #                system_folder,
            #                "{}/{}/{}-{},{}{}.json".format(root_folder,output_folder,
            #                                                   ate, embedding_setting,
            #                                                   graph_setting, k),
            #                pnl,
            #                "{},{}{}".format(embedding_setting, graph_setting, k),
            #                jate_outfolder_per_file,
            #                settings, personalization_seed)

    return settings
