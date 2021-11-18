from pyserini.index import IndexReader
from tqdm import tqdm
import numpy as np

class Ranker(object):
    '''
    The base class for ranking functions. Specific ranking functions should
    extend the score() function, which returns the relevance of a particular 
    document for a given query.
    '''
    
    def __init__(self, index_reader):
        self.index_reader = index_reader

    def score(query, doc):        
        '''
        Returns the score for how relevant this document is to the provided query.
        Query is a tokenized list of query terms and doc_id is the identifier
        of the document in the index should be scored for this query.
        '''
        
        rank_score = 0
        return rank_score


class PivotedLengthNormalizationRanker(Ranker):
    
    def __init__(self, index_reader):
        super(PivotedLengthNormalizationRanker, self).__init__(index_reader)
        
        self.index_reader = index_reader
        self.stats = index_reader.stats()
        self.n_docs = self.stats['documents']
        self.avg_dl = self.stats['total_terms'] / self.n_docs
        self.docvec_cache = {}
        self.freq_cache = {}

    def score(self, query, doc_id, b=0.5):
        '''
        Scores the relevance of the document for the provided query using the
        Pivoted Length Normalization ranking method. Query is a tokenized list
        of query terms and doc_id is a numeric identifier of which document in the
        index should be scored for this query.

        '''
        rank_score = 0
        
        if doc_id in self.docvec_cache:
            doc_vector = self.docvec_cache[doc_id]
        else:
            doc_vector = self.index_reader.get_document_vector(doc_id)
            self.docvec_cache[doc_id] = doc_vector
        
        doc_query_set = set(query).intersection(doc_vector.keys())  # q∩d
        doc_length = sum(doc_vector.values())
        
        for term in doc_query_set:
            if term in self.freq_cache:
                df, cf = self.freq_cache[term]
            else:
                df, cf = self.index_reader.get_term_counts(term)  # df(w)
                self.freq_cache[term] = [df, cf]
            
            qtf = query.count(term)  # c(w, q)
            tf = doc_vector[term]  # c(w, d)
            norm_tf = (1 + np.log(1 + np.log(tf))) / (1 - b + (b * doc_length / self.avg_dl))
            idf = np.log((self.n_docs + 1) / df)
            term_score = qtf * norm_tf * idf
            
            rank_score += term_score
    
        return rank_score
    

class BM25Ranker(Ranker):

    def __init__(self, index_reader):
        super(BM25Ranker, self).__init__(index_reader)
        
        self.index_reader = index_reader
        self.stats = index_reader.stats()
        self.n_docs = self.stats['documents']
        self.avg_dl = self.stats['total_terms'] / self.n_docs
        self.docvec_cache = {}
        self.freq_cache = {}

    def score(self, query, doc_id, k1=1.2, b=0.3, k3=1.5):
        '''
        Scores the relevance of the document for the provided query using the
        BM25 ranking method. Query is a tokenized list of query terms and doc_id
        is a numeric identifier of which document in the index should be scored
        for this query.
        '''
        rank_score = 0
        
        if doc_id in self.docvec_cache:
            doc_vector = self.docvec_cache[doc_id]
        else:
            doc_vector = self.index_reader.get_document_vector(doc_id)
            self.docvec_cache[doc_id] = doc_vector
        
        doc_query_set = set(query).intersection(doc_vector.keys())
        doc_length = sum(doc_vector.values())
        
        for term in doc_query_set:
            if term in self.freq_cache:
                df, cf = self.freq_cache[term]
            else:
                df, cf = self.index_reader.get_term_counts(term)
                self.freq_cache[term] = [df, cf]
            
            qtf = query.count(term)
            tf = doc_vector[term]
            idf = np.log((self.n_docs - df + 0.5) / (df + 0.5))
            norm_tf = ((k1 + 1) * tf) / (k1 * (1 - b + (b * doc_length / self.avg_dl)) + tf)
            norm_qtf = ((k3 + 1) * qtf) / (k3 + qtf)
            term_score = idf * norm_tf * norm_qtf
            
            rank_score += term_score

        return rank_score

    
class CustomRanker(Ranker):
    
    def __init__(self, index_reader):
        super(CustomRanker, self).__init__(index_reader)

        self.index_reader = index_reader
        self.stats = index_reader.stats()
        self.n_docs = self.stats['documents']
        self.n_terms = self.stats['total_terms']
        self.avg_dl = self.n_terms / self.n_docs
        self.docvec_cache = {}
        self.freq_cache = {}
        self.positions_cache = {}

    def score(self, query, doc_id, lmd=0.6, sm=0.2):
        '''
        Scores the relevance of the document for the provided query using a
        custom ranking method. Query is a tokenized list of query terms and doc_id
        is a numeric identifier of which document in the index should be scored
        for this query.
        '''
        rank_score = 0
        
        
        if doc_id in self.docvec_cache:
            doc_vector = self.docvec_cache[doc_id]
        else:
            doc_vector = self.index_reader.get_document_vector(doc_id)
            self.docvec_cache[doc_id] = doc_vector
        
        if doc_id in self.positions_cache:
            doc_positions = self.positions_cache[doc_id]
        else:
            doc_positions = self.index_reader.get_term_positions(doc_id)
            self.positions_cache[doc_id] = doc_positions
        
        doc_query_set = set(query).intersection(doc_vector.keys())
        doc_length = sum(doc_vector.values())
        query_length = len(query)
        query_position = {k: v for v, k in enumerate(query)}
        
        for term in doc_query_set:
            if term in self.freq_cache:
                df, cf = self.freq_cache[term]
            else:
                df, cf = self.index_reader.get_term_counts(term)
                self.freq_cache[term] = [df, cf]
        
            qtf = query.count(term)
            tf = doc_vector[term]
            # term relative position
            trp_q = query_position[term] / query_length  # give importance to end of query
            trp_d = np.log(np.log(doc_length + 1) / np.log(np.mean(doc_positions[term]) + 1))  # give importance to start of document
            # collection importance
            ci = (cf / self.n_terms) * (self.n_docs / df)
            idf = np.log(self.n_docs / df)
            
            term_score = lmd * (np.log(tf) / (qtf * trp_q * trp_d + sm)) + (1 - lmd) * (ci * idf + sm)
            
            rank_score += term_score

        return rank_score
