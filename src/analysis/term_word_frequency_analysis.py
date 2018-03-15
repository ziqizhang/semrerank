# given ttf for candidate terms, calculate ttf for non-stop words, then find out the candidate terms whose composing words are all
# below a certain ttf threshold, save them onto a list
import csv
from collections import OrderedDict

import numpy as np
import os

from nltk.corpus import stopwords

import graph.semrerank_doc_graph as srr
import util.utils as ut
from exp import exp_loader_doc_graph

stop = stopwords.words('english')


# given ttf of candidate terms in a JATE2.0 json format, convert them to word tf in a csv format.
def calc_word_freq(ttf_term_json, ttf_word_out_file):
    term_ttf_scores = {c[0]: c[1] for c in srr.jate_terms_iterator(ttf_term_json)}
    word_freq = {}

    for t, ttf in term_ttf_scores.items():
        parts = t.split(" ")

        for p in parts:
            p = exp_loader_doc_graph.lemmatizer.lemmatize(p).strip().lower()
            if p in stop or len(p) < 2:
                continue

            if p in word_freq.keys():
                word_freq[p] += ttf
            else:
                word_freq[p] = ttf

    sorted_w_ttf = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    with open(ttf_word_out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for tuple in sorted_w_ttf:
            writer.writerow([tuple[0], tuple[1]])


# given a candidate list of terms, and a list of word tf on the same corpus, find candidate terms whose composing words
# have a word tf that is below a threshold. the optional gs_file will also check if that term is a real term. NOTE that
# this gs file must contain real terms that are already pre-processed by JATE!
def find_terms_with_infrequent_words(ttf_term_json, ttf_word_csv_file, min_ttf_word, out_file,
                                     gs_file=None):
    term_ttf_scores = {c[0]: c[1] for c in srr.jate_terms_iterator(ttf_term_json)}
    word_ttf_scores = {}
    with open(ttf_word_csv_file, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            word_ttf_scores[row[0]] = row[1]
    gs_terms = []
    if gs_file is not None:
        with open(gs_file) as f:
            terms = f.readlines()
            for t in terms:
                t = exp_loader_doc_graph.lemmatizer.lemmatize(t).strip().lower()
                if len(t) >= 2:
                    gs_terms.append(t)

    selected_terms = []
    for t, ttf in term_ttf_scores.items():
        parts = t.split(" ")

        all_words_infrequent = True
        for p in parts:
            p = exp_loader_doc_graph.lemmatizer.lemmatize(p).strip().lower()
            if p in stop or len(p) < 2:
                continue

            if p in word_ttf_scores.keys():
                wtf = word_ttf_scores[p]
                if float(wtf) >= min_ttf_word:
                    all_words_infrequent = False
                    break

        if all_words_infrequent:
            if gs_terms is not None and t in gs_terms:
                selected_terms.append(t)
            elif gs_terms is None:
                selected_terms.append(t)

    selected_terms = sorted(selected_terms)
    with open(out_file, 'w') as the_file:
        for t in selected_terms:
            the_file.write(t + '\n')


def calc_avg_word_freq(term, word_ttf_scores):
    parts = term.split(" ")

    count = 0
    total_freq = 0
    for p in parts:
        p = exp_loader_doc_graph.lemmatizer.lemmatize(p).strip().lower()
        if p in stop or len(p) < 2:
            continue

        count += 1
        if p in word_ttf_scores.keys():
            wtf = word_ttf_scores[p]
            total_freq += float(wtf)

    if count == 0:
        return 0
    return total_freq / count


# given the ranked list of terms by a base ATE, and also the list by srr, and a reference list of terms to be searched,
# for each term in the reference list, check its rank in srr, base ATE, calculate advancement as % on the entire list, and
# save the distribution over different % advancements
def calc_movement_stats(base_ate_outlist_json, srr_ate_outlist_json, term_filter_list_file, word_ttf_scores: dict):
    with open(term_filter_list_file) as f:
        filter_terms = f.readlines()
    base_ate_scores = {c[0]: c[1] for c in srr.jate_terms_iterator(base_ate_outlist_json)}
    sorted_base_ate = sorted(base_ate_scores, key=base_ate_scores.get, reverse=True)
    ssr_ate_scores = {c[0]: c[1] for c in ut.semrerank_json_reader(srr_ate_outlist_json)}
    sorted_ssr_ate = sorted(ssr_ate_scores, key=ssr_ate_scores.get, reverse=True)

    distribution = {}
    sum_avg_word_freq = {}  # key: movement range; value:sum of the average word freq for each term belong to that range
    for adv in np.arange(-1.0, 1.0, 0.05):
        distribution[format(adv, '.2f')] = 0
        sum_avg_word_freq[format(adv, '.2f')] = 0

    count_advances = 0
    count_advanced_percentages = 0
    count_drops = 0
    count_drop_percentages = 0
    for f_t in filter_terms:
        f_t = f_t.strip()
        if f_t not in sorted_base_ate:
            continue

        avg_word_freq = calc_avg_word_freq(f_t, word_ttf_scores)
        ate_index = sorted_base_ate.index(f_t)
        ssr_index = sorted_ssr_ate.index(f_t)

        advance = (ate_index - ssr_index) / len(sorted_base_ate)

        if advance > 0:
            count_advanced_percentages += advance
            count_advances += 1
        elif advance < 0:
            count_drop_percentages += advance
            count_drops += 1

        if advance >= -1.0 and advance < -0.95:
            distribution['-1.0'] += 1
            sum_avg_word_freq['-1.0'] += avg_word_freq
        elif advance >= -0.95 and advance < -0.9:
            distribution['-0.95'] += 1
            sum_avg_word_freq['-0.95'] += avg_word_freq
        elif advance >= -0.9 and advance < -0.85:
            distribution['-0.90'] += 1
            sum_avg_word_freq['-0.90'] += avg_word_freq
        elif advance >= -0.85 and advance < -0.8:
            distribution['-0.85'] += 1
            sum_avg_word_freq['-0.85'] += avg_word_freq
        elif advance >= -0.8 and advance < -0.75:
            distribution['-0.80'] += 1
            sum_avg_word_freq['-0.80'] += avg_word_freq
        elif advance >= -0.75 and advance < -0.7:
            distribution['-0.75'] += 1
            sum_avg_word_freq['-0.75'] += avg_word_freq
        elif advance >= -0.70 and advance < -0.65:
            distribution['-0.70'] += 1
            sum_avg_word_freq['-0.70'] += avg_word_freq
        elif advance >= -0.65 and advance < -0.6:
            distribution['-0.65'] += 1
            sum_avg_word_freq['-0.65'] += avg_word_freq
        elif advance >= -0.6 and advance < -0.55:
            distribution['-0.60'] += 1
            sum_avg_word_freq['-0.60'] += avg_word_freq
        elif advance >= -0.55 and advance < -0.5:
            distribution['-0.55'] += 1
            sum_avg_word_freq['-0.55'] += avg_word_freq
        elif advance >= -0.5 and advance < -0.45:
            distribution['-0.50'] += 1
            sum_avg_word_freq['-0.50'] += avg_word_freq
        elif advance >= -0.45 and advance < -0.4:
            distribution['-0.45'] += 1
            sum_avg_word_freq['-0.45'] += avg_word_freq
        elif advance >= -0.4 and advance < -0.35:
            distribution['-0.40'] += 1
            sum_avg_word_freq['-0.40'] += avg_word_freq
        elif advance >= -0.35 and advance < -0.3:
            distribution['-0.35'] += 1
            sum_avg_word_freq['-0.35'] += avg_word_freq
        elif advance >= -0.3 and advance < -0.25:
            distribution['-0.30'] += 1
            sum_avg_word_freq['-0.30'] += avg_word_freq
        elif advance >= -0.25 and advance < -0.2:
            distribution['-0.25'] += 1
            sum_avg_word_freq['-0.25'] += avg_word_freq
        elif advance >= -0.20 and advance < -0.15:
            distribution['-0.20'] += 1
            sum_avg_word_freq['-0.20'] += avg_word_freq
        elif advance >= -0.15 and advance < -0.1:
            distribution['-0.15'] += 1
            sum_avg_word_freq['-0.15'] += avg_word_freq
        elif advance >= -1.0 and advance < -0.05:
            distribution['-0.10'] += 1
            sum_avg_word_freq['-0.10'] += avg_word_freq
        elif advance >= -0.05 and advance < 0:
            distribution['-0.05'] += 1
            sum_avg_word_freq['-0.05'] += avg_word_freq
        elif advance == 0:
            distribution['0.00'] += 1
            sum_avg_word_freq['0.00'] += avg_word_freq
        elif advance > 0 and advance < 0.05:
            distribution['0.05'] += 1
            sum_avg_word_freq['0.05'] += avg_word_freq
        elif advance >= 0.05 and advance < 0.1:
            distribution['0.10'] += 1
            sum_avg_word_freq['0.10'] += avg_word_freq
        elif advance >= 0.1 and advance < 0.15:
            distribution['0.15'] += 1
            sum_avg_word_freq['0.15'] += avg_word_freq
        elif advance >= 0.15 and advance < 0.2:
            distribution['0.20'] += 1
            sum_avg_word_freq['0.20'] += avg_word_freq
        elif advance >= 0.2 and advance < 0.25:
            distribution['0.25'] += 1
            sum_avg_word_freq['0.25'] += avg_word_freq
        elif advance >= 0.25 and advance < 0.3:
            distribution['0.30'] += 1
            sum_avg_word_freq['0.30'] += avg_word_freq
        elif advance >= 0.35 and advance < 0.4:
            distribution['0.35'] += 1
            sum_avg_word_freq['0.35'] += avg_word_freq
        elif advance >= 0.4 and advance < 0.45:
            distribution['0.40'] += 1
            sum_avg_word_freq['0.40'] += avg_word_freq
        elif advance >= 0.45 and advance < 0.5:
            distribution['0.45'] += 1
            sum_avg_word_freq['0.45'] += avg_word_freq
        elif advance >= 0.5 and advance < 0.55:
            distribution['0.50'] += 1
            sum_avg_word_freq['0.50'] += avg_word_freq
        elif advance >= 0.55 and advance < 0.6:
            distribution['0.55'] += 1
            sum_avg_word_freq['0.55'] += avg_word_freq
        elif advance >= 0.6 and advance < 0.65:
            distribution['0.60'] += 1
            sum_avg_word_freq['0.60'] += avg_word_freq
        elif advance >= 0.65 and advance < 0.7:
            distribution['0.65'] += 1
            sum_avg_word_freq['0.65'] += avg_word_freq
        elif advance >= 0.7 and advance < 0.75:
            distribution['0.70'] += 1
            sum_avg_word_freq['0.70'] += avg_word_freq
        elif advance >= 0.75 and advance < 0.8:
            distribution['0.75'] += 1
            sum_avg_word_freq['0.75'] += avg_word_freq
        elif advance >= 0.8 and advance < 0.85:
            distribution['0.80'] += 1
            sum_avg_word_freq['0.80'] += avg_word_freq
        elif advance >= 0.85 and advance < 0.9:
            distribution['0.85'] += 1
            sum_avg_word_freq['0.85'] += avg_word_freq
        elif advance >= 0.9 and advance < 0.95:
            distribution['0.90'] += 1
            sum_avg_word_freq['0.90'] += avg_word_freq
        elif advance >= 0.95 and advance <= 1.0:
            distribution['0.95'] += 1
            sum_avg_word_freq['0.95'] += avg_word_freq

    avg_adv = 0
    avg_drop = 0
    if count_advances > 0:
        avg_adv = count_advanced_percentages / count_advances
    if count_drops > 0:
        avg_drop = count_drop_percentages / count_drops
    print("avg advance,{},avg drop,{}".format(avg_adv,
                                              avg_drop))

    avg_word_freq = {}
    for mv_range, sum_avg_w_f in sum_avg_word_freq.items():
        samples = distribution[mv_range]
        if samples>0:
            avg_word_freq[mv_range] = sum_avg_w_f/samples
        else:
            avg_word_freq[mv_range]=0

    return OrderedDict(sorted(distribution.items())), OrderedDict(sorted(avg_word_freq.items()))


# given a folder of base ate output term lists and a folder of ssr ate output term lists, for each
# pair of base ate-ssr ate, use the method above (calc_avg_advancement) to prepare advancement stats and
# save to a csv file
def calc_movement_stats_batch(base_ate_outlist_folder, srr_ate_outlist_folder, term_filter_list_file,
                              ttf_word_csv_file,
                              out_file_csv):
    word_ttf_scores = {}
    with open(ttf_word_csv_file, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            word_ttf_scores[row[0]] = row[1]

    # create a map between base ate and its ssr enhanced output
    ate_to_ssroutfile = {}
    for ssr_f in os.listdir(srr_ate_outlist_folder):
        pos = ssr_f.find('-')
        base_ate_name = ssr_f[0:pos]
        ate_to_ssroutfile[base_ate_name] = ssr_f

    with open(out_file_csv, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                       quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for base_ate_json in os.listdir(base_ate_outlist_folder):
            base_ate_name = base_ate_json
            pos = base_ate_name.find('.')
            base_ate_name = base_ate_name[0:pos]
            if base_ate_name not in ate_to_ssroutfile:
                continue
            ssr_ate_json = ate_to_ssroutfile[base_ate_name]
            stats = calc_movement_stats(base_ate_outlist_folder + "/" + base_ate_json,
                                        srr_ate_outlist_folder + "/" + ssr_ate_json,
                                        term_filter_list_file, word_ttf_scores)[1]
            values = []
            for k, v in stats.items():
                values.append(v)
                if k == '-1.00':
                    values.reverse()
            values.insert(0, base_ate_json)
            w.writerow(values)


if __name__ == "__main__":
    # >>> step ONE for converting ttf to word tf in a corpus
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
    # ttf_term_json_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/ttf.json"
    # ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/ttf_words.csv"
    # calc_word_freq(ttf_term_json_file,ttf_word_out_file)

    # >>> step TWO for finding candidate terms whose composing words are all below a min ttf
    min_wtf = 5
    # gs_terms_file="/home/zz/Work/data/jate_data/acl-rd-corpus-2.0/acl-rd-ver2-gs-terms_pruned.txt"
    # selected_rare_terms_out_file="/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/t_with_infrequent_words.txt"
    # selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/t_with_infrequent_words.txt"
    # gs_terms_file = "/home/zz/Work/data/jate_data/genia_gs/concept/genia_gs_terms_v2_pruned.txt"
    # selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia/t_with_infrequent_words.txt"
    # selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia_atr4s/t_with_infrequent_words.txt"
    # gs_terms_file = "/home/zz/Work/data/jate_data/ttc/gs-en-mobile-technology_pruned.txt"
    # selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile/t_with_infrequent_words.txt"
    # selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile_atr4s/t_with_infrequent_words.txt"
    # gs_terms_file = "/home/zz/Work/data/jate_data/ttc/gs-en-windenergy_pruned.txt"
    # selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind/t_with_infrequent_words.txt"
    # selected_rare_terms_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/t_with_infrequent_words.txt"

    # find_terms_with_infrequent_words(ttf_term_json_file,ttf_word_out_file,min_wtf,selected_rare_terms_out_file,
    #                                  gs_terms_file)

    # >>> step THREE for calculating advancement stats by semrerank
    #base_ate_outfolder="/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/min1"
    #ssr_ate_outfolder="/home/zz/Work/data/semrerank/output_aclv2_atr4s/z=100"
    #ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/ttf_words.csv"
    #filter_terms_file = "/home/zz/Work/data/jate_data/acl-rd-corpus-2.0/acl-rd-ver2-gs-terms_pruned.txt"
#   filter_terms_file="/home/zz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/t_with_infrequent_words.txt"
    #output_csv="/home/zz/Work/data/semrerank/advance_stats_aclv2.csv"
    #calc_movement_stats_batch(base_ate_outfolder, ssr_ate_outfolder, filter_terms_file,ttf_word_out_file,output_csv)

    # base_ate_outfolder = "/home/zz/Work/data/semrerank/jate_lrec2016/genia_atr4s/min1"
    # ssr_ate_outfolder = "/home/zz/Work/data/semrerank/output_genia_atr4s/z=100"
    # ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia_atr4s/ttf_words.csv"
    # #filter_terms_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia_atr4s/t_with_infrequent_words.txt"
    # filter_terms_file = "/home/zz/Work/data/jate_data/genia_gs/concept/genia_gs_terms_v2_pruned.txt"
    # output_csv = "/home/zz/Work/data/semrerank/advance_stats_genia.csv"
    # calc_movement_stats_batch(base_ate_outfolder, ssr_ate_outfolder, filter_terms_file, ttf_word_out_file,output_csv)

    # base_ate_outfolder = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile_atr4s/min1"
    # ssr_ate_outfolder = "/home/zz/Work/data/semrerank/output_ttcm_atr4s/z=100"
    # ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_mobile_atr4s/ttf_words.csv"
    # #filter_terms_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia_atr4s/t_with_infrequent_words.txt"
    # filter_terms_file = "/home/zz/Work/data/jate_data/ttc/gs-en-mobile-technology_pruned.txt"
    # output_csv = "/home/zz/Work/data/semrerank/advance_stats_ttcm.csv"
    # calc_movement_stats_batch(base_ate_outfolder, ssr_ate_outfolder, filter_terms_file, ttf_word_out_file,output_csv)

    base_ate_outfolder = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/min1"
    ssr_ate_outfolder = "/home/zz/Work/data/semrerank/output_ttcw_atr4s/z=100"
    ttf_word_out_file = "/home/zz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/ttf_words.csv"
    #filter_terms_file = "/home/zz/Work/data/semrerank/jate_lrec2016/genia_atr4s/t_with_infrequent_words.txt"
    filter_terms_file = "/home/zz/Work/data/jate_data/ttc/gs-en-windenergy_pruned.txt"
    output_csv = "/home/zz/Work/data/semrerank/advance_stats_ttcw.csv"
    calc_movement_stats_batch(base_ate_outfolder, ssr_ate_outfolder, filter_terms_file, ttf_word_out_file,output_csv)

