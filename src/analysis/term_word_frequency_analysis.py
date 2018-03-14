
#given ttf for candidate terms, calculate ttf for non-stop words, then find out the candidate terms whose composing words are all
#below a certain ttf threshold, save them onto a list
import csv

from nltk.corpus import stopwords

import graph.semrerank_doc_graph as srr
from exp import exp_loader_doc_graph

stop = stopwords.words('english')

#given ttf of candidate terms in a JATE2.0 json format, convert them to word tf in a csv format.
def calc_word_freq(ttf_term_json, ttf_word_out_file):
    term_ttf_scores = {c[0]: c[1] for c in srr.jate_terms_iterator(ttf_term_json)}
    word_freq={}

    for t, ttf in term_ttf_scores.items():
        parts = t.split(" ")

        for p in parts:
            p = exp_loader_doc_graph.lemmatizer.lemmatize(p).strip().lower()
            if p in stop or len(p)<2:
                continue

            if p in word_freq.keys():
                word_freq[p]+=ttf
            else:
                word_freq[p]=ttf

    sorted_w_ttf = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)

    with open(ttf_word_out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for tuple in sorted_w_ttf:
            writer.writerow([tuple[0],tuple[1]])


#given a candidate list of terms, and a list of word tf on the same corpus, find candidate terms whose composing words
#have a word tf that is below a threshold. the optional gs_file will also check if that term is a real term. NOTE that
#this gs file must contain real terms that are already pre-processed by JATE!
def find_terms_with_infrequent_words(ttf_term_json, ttf_word_csv_file, min_ttf_word, out_file,
                                     gs_file=None):
    term_ttf_scores = {c[0]: c[1] for c in srr.jate_terms_iterator(ttf_term_json)}
    word_ttf_scores={}
    with open(ttf_word_csv_file, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            word_ttf_scores[row[0]]=row[1]
    gs_terms=[]
    if gs_file is not None:
        with open(gs_file) as f:
            terms = f.readlines()
            for t in terms:
                t=exp_loader_doc_graph.lemmatizer.lemmatize(t).strip().lower()
                if len(t)>=2:
                    gs_terms.append(t)

    selected_terms=[]
    for t, ttf in term_ttf_scores.items():
        parts = t.split(" ")

        all_words_infrequent=True
        for p in parts:
            p = exp_loader_doc_graph.lemmatizer.lemmatize(p).strip().lower()
            if p in stop or len(p) < 2:
                continue

            if p in word_ttf_scores.keys():
                wtf = word_ttf_scores[p]
                if float(wtf)>=min_ttf_word:
                    all_words_infrequent=False
                    break

        if all_words_infrequent:
            if gs_terms is not None and t in gs_terms:
                selected_terms.append(t)
            elif gs_terms is None:
                selected_terms.append(t)

    selected_terms=sorted(selected_terms)
    with open(out_file, 'w') as the_file:
        for t in selected_terms:
            the_file.write(t+'\n')

if __name__ == "__main__":

    #>>> code for converting ttf to word tf in a corpus
    # ttf_term_json_file="/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/ttf.json"
    # ttf_word_out_file="/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/ttf_words.csv"
    # ttf_term_json_file = "/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/ttf.json"
    # ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/ttf_words.csv"
    # ttf_term_json_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia/ttf.json"
    # ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia/ttf_words.csv"
    # ttf_term_json_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia_atr4s/ttf.json"
    # ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia_atr4s/ttf_words.csv"
    # ttf_term_json_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile/ttf.json"
    # ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile/ttf_words.csv"
    # ttf_term_json_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile_atr4s/ttf.json"
    # ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile_atr4s/ttf_words.csv"
    # ttf_term_json_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind/ttf.json"
    # ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind/ttf_words.csv"
    ttf_term_json_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/ttf.json"
    ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/ttf_words.csv"
    # calc_word_freq(ttf_term_json_file,ttf_word_out_file)

    
    # >>> code for finding candidate terms whose composing words are all below a min ttf
    min_wtf=5
    # gs_terms_file="/home/zz/Work/data/jate_data/acl-rd-corpus-2.0/acl-rd-ver2-gs-terms_pruned.txt"
    # selected_rare_terms_out_file="/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/t_with_infrequent_words.txt"
    # selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/t_with_infrequent_words.txt"
    #gs_terms_file = "/home/zz/Work/data/jate_data/genia_gs/concept/genia_gs_terms_v2_pruned.txt"
    #selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia/t_with_infrequent_words.txt"
    #selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia_atr4s/t_with_infrequent_words.txt"
    #gs_terms_file = "/home/zz/Work/data/jate_data/ttc/gs-en-mobile-technology_pruned.txt"
    #selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile/t_with_infrequent_words.txt"
    #selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile_atr4s/t_with_infrequent_words.txt"
    gs_terms_file = "/home/zz/Work/data/jate_data/ttc/gs-en-windenergy_pruned.txt"
    #selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind/t_with_infrequent_words.txt"
    selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/t_with_infrequent_words.txt"

    find_terms_with_infrequent_words(ttf_term_json_file,ttf_word_out_file,min_wtf,selected_rare_terms_out_file,
                                     gs_terms_file)