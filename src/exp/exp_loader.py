import numpy as np
import os
#process = psutil.Process(os.getpid())

DIRECTED_GRAPH=False
GRAPH_PRUNING=False
SCORING_METHOD=1  #1=multiple base jate score; 2=add base jate score; 3=discard jate base score

MAX_PERCENTAGE_OF_VERTICES_FOR_PERSONALIZATION_INIT=0.5

def create_setting(word2vec_model, jate_out_terms, topN_similar, similarity_threshold, jate_term_max_n,
        pagerank_out_file, final_out_file, topN_personalized, exp_id, appended_list):
    props=(word2vec_model, jate_out_terms, topN_similar, similarity_threshold, jate_term_max_n,
        pagerank_out_file, final_out_file, topN_personalized, exp_id)
    appended_list.append(props)


def create_settings():
    settings=list()
    root_folder="/home/zqz/Work/data/semrerank"
    jate_terms_folder="/home/zqz/Work/data/semrerank/jate_lrec2016/genia/min2"
    #root_folder="/home/zqz/Work/data/semrerank"
    #jate_terms_folder="/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd/min5"
    #print(process.memory_info().rss)

    for file in os.listdir(jate_terms_folder):
        for k in np.arange(0.9, 0.5, -0.1):
            embedding_setting="em_g-uni-sg-100-w3-m2"
            topN=500
            base=os.path.basename(file)
            ate=os.path.splitext(base)[0]

            k=round(k,1)
            graph_setting="g_dn-pn-top{}-pnl0-t".format(topN)
            create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
                       jate_terms_folder+"/"+str(file),
                       topN,
                       k,
                       5,
                       "{}/graph/{}_{}{}.pageranks".format(root_folder, embedding_setting,
                                                           graph_setting, k),
                       "{}/output/{}-{},{}{}.json".format(root_folder,
                                                                     ate,embedding_setting,
                                                                     graph_setting,k),
                        None,
                       "{},{}{}".format(embedding_setting, graph_setting, k),
                       settings)
            #print(process.memory_info().rss)

            # graph_setting="g_dn-pn-top{}-pnl2.20-t".format(topN)
            # create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
            #            jate_terms_folder+"/"+str(file),
            #            topN,
            #            k,
            #            5,
            #            "{}/graph/{}_{}{}.pageranks".format(root_folder, embedding_setting,
            #                                                graph_setting, k),
            #            "{}/output/{}-{},{}{}.json".format(root_folder,
            #                                                          ate,embedding_setting,
            #                                                          graph_setting,k),
            #             20,
            #            "{},{}{}".format(embedding_setting, graph_setting, k),
            #            settings)
            # print(process.memory_info().rss)
            #
            # #personalized pagerank
            # pnl=50
            # graph_setting="g_dn-pn-top{}-pnl2.{}-t".format(topN,pnl)
            # create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
            #            jate_terms_folder+"/"+str(file),
            #            topN,
            #            k,
            #            5,
            #            "{}/graph/{}_{}{}.pageranks".format(root_folder, embedding_setting,
            #                graph_setting, k),
            #            "{}/output/{}-{},{}{}.json".format(root_folder,ate,embedding_setting,
            #                                                          graph_setting, k),
            #             pnl,
            #            "{},{}{}".format(embedding_setting, graph_setting, k),
            #            settings)
            # print(process.memory_info().rss)
            #
            # pnl=100
            # graph_setting="g_dn-pn-top{}-pnl2.{}-t".format(topN,pnl)
            # create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
            #            jate_terms_folder+"/"+str(file),
            #            topN,
            #            k,
            #            5,
            #            "{}/graph/{}_{}{}.pageranks".format(root_folder, embedding_setting,
            #                graph_setting, k),
            #            "{}/output/{}-{},{}{}.json".format(root_folder,ate,embedding_setting,
            #                                                          graph_setting, k),
            #             pnl,
            #            "{},{}{}".format(embedding_setting, graph_setting, k),
            #            settings)
            # print(process.memory_info().rss)
            #
            # pnl=200
            # graph_setting="g_dn-pn-top{}-pnl2.{}-t".format(topN,pnl)
            # create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
            #            jate_terms_folder+"/"+str(file),
            #            topN,
            #            k,
            #            5,
            #            "{}/graph/{}_{}{}.pageranks".format(root_folder, embedding_setting,
            #                graph_setting, k),
            #            "{}/output/{}-{},{}{}.json".format(root_folder,ate,embedding_setting,
            #                                                          graph_setting, k),
            #             pnl,
            #            "{},{}{}".format(embedding_setting, graph_setting, k),
            #            settings)
            # print(process.memory_info().rss)
            #
            # pnl=500
            # graph_setting="g_dn-pn-top{}-pnl2.{}-t".format(topN,pnl)
            # create_setting("{}/embeddings/{}.model".format(root_folder, embedding_setting),
            #            jate_terms_folder+"/"+str(file),
            #            topN,
            #            k,
            #            5,
            #            "{}/graph/{}_{}{}.pageranks".format(root_folder, embedding_setting,
            #                graph_setting, k),
            #            "{}/output/{}-{},{}{}.json".format(root_folder,ate,embedding_setting,
            #                                                          graph_setting, k),
            #             pnl,
            #            "{},{}{}".format(embedding_setting, graph_setting, k),
            #            settings)
            # print(process.memory_info().rss)

    return settings

