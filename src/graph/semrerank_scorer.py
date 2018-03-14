
#given a candidate term, and a list of component word/phrases found on the graph, return the optimum subset of the components
#that covers the term
from exp import exp_loader_doc_graph
from util import utils
from nltk.corpus import stopwords


def find_components(key, value):
    return value


class SemReRankScorer(object):
    #pageranks: page rank scores of composite words/phrases
    #jate_term_map: a dictionary of pairs (term, composite words/phrases found in the graph). this should be produced by semrerank
    #base_scores: can be none. If NOT none, should be a dictionary of pars (term-original ATE score)
    def __init__(self, pageranks, jate_term_map, base_scores):
        self.pageranks = pageranks
        self.base_scores=base_scores
        if exp_loader_doc_graph.SCORING_NORMALIZE_JATE_SCORE:
            self.base_scores=utils.normalize(base_scores)
        self.jate_term_map=jate_term_map

    def __iter__(self):
        for key,value in self.jate_term_map.items():
            components = find_components(key, value)

            term_tokens=len(key.split())

            totalScore=0.0
            totalNonZeroComponents=0.0
            for c in components:
                if c in self.pageranks.keys():
                    if exp_loader_doc_graph.REMOVE_STOPWORDS and c not in stopwords.words('english'):
                        totalScore=totalScore+self.pageranks[c]
                        totalNonZeroComponents=totalNonZeroComponents+1.0
                    elif not exp_loader_doc_graph.REMOVE_STOPWORDS:
                        totalScore=totalScore+self.pageranks[c]
                        totalNonZeroComponents=totalNonZeroComponents+1.0

            avg_rank=0.0
            if totalScore>0 and totalNonZeroComponents>0:
                #rescale=totalScore/totalNonZeroComponents
                avg_rank=totalScore/term_tokens

            if self.base_scores is None: #no base scores provided, so we do not modify the scores but only using page rank scores
                yield {"string": key, "score-mult": totalScore,  "score-add": totalScore,"rank":totalScore}
            else:
                if key not in self.base_scores:
                    print("\t\t warning:: this term is not in the original ATE list of terms:{}".format(key))
                    continue
                bs = self.base_scores[key]

                #if(exp_loader_doc_based.SCORING_METHOD==1):
                ns_mult = (avg_rank+1)*bs
                #else:
                ns_add=avg_rank+bs
                yield {"string": key, "score-mult": ns_mult, "score-add":ns_add,"rank":(avg_rank+1)}

