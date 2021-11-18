from rankers import PivotedLengthNormalizationRanker, BM25Ranker, CustomRanker
from pyserini.index import IndexReader
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
import sys
import json


def rank_query(ranker, query):
    '''
    Prints the relevance scores of the top retrieved documents.
    '''
    doc_score = {}
    for i in tqdm(doc_list):
        score = 0
        try:
            score = ranker.score(query, str(i))
        except:
            pass
        doc_score[i] = score
    
    return doc_score
        

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("usage: python main.py ranker path/to/index_file path/to/queries")
        exit(1)

    # NOTE: You should already have used pyserini to generate the index files
    # before calling main
    index_fname = sys.argv[2]
    index = "_" + index_fname.split("/")[1] if "trec_covid" not in index_fname else ""
    index_reader = IndexReader(index_fname)  # Reading the indexes
    doc_list = pd.read_csv(f"documents{index}.csv")["DocumentId"]

    # Print some basic stats
    print("Loaded dataset with the following statistics: " + str(index_reader.stats()))

    # NOTE: You can extend this code to have the program read a list of queries
    # and generate rankings for each.
    
    print("Initializing Ranker...")
    ranker_option = sys.argv[1]
    # Choose which ranker class you want to use
    if ranker_option == "plnr":
        ranker = PivotedLengthNormalizationRanker(index_reader)
    elif ranker_option == "bm25":
        ranker = BM25Ranker(index_reader)
    elif ranker_option == "custom":
        ranker = CustomRanker(index_reader)
    else:
        print("ranker options: plnr, bm25, custom")
        exit(1)
    
    print("Loading Queries...")
    queries = pd.read_csv(sys.argv[3]).iloc[1:]
    stop = set(stopwords.words('english') + list(string.punctuation))

    with open(f"ranking_{ranker_option}.txt", "w") as f:
        f.write("queryid,DocumentId\n")
    
    for i, row in queries.iterrows():
        print(f"Evaluating Query #{i}/#{queries.shape[0]}...")
        query = row["Query Description"].lower()
        query_stem = []
        for w in word_tokenize(query):
            if w not in stop:
                query_stem.extend(index_reader.analyze(w))
        doc_score = rank_query(ranker, query_stem)
        doc_ranked = sorted(doc_score, key=doc_score.get, reverse=True)
        with open(f"ranking_{ranker_option}.txt", "a") as f:
            for doc_id in doc_ranked:
                f.write(f"{row['QueryId']},{doc_id}\n")
        
