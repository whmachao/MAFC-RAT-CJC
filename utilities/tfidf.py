#!/usr/bin/env python

"""The simplest TF-IDF library imaginable.

Add your documents as two-element lists `[docname,
[list_of_words_in_the_document]]` with `addDocument(docname, list_of_words)`.
Get a list of all the `[docname, similarity_score]` pairs relative to a
document by calling `similarities([list_of_words])`.

See the README for a usage example.

"""

import sys
import os
import utilities.Constants as Constants


class TfIdf:
    def __init__(self):
        self.weighted = False
        self.documents = []
        self.corpus_dict = {}

    def add_document(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0

        # normalizing the dictionary
        length = float(len(list_of_words))
        for k in doc_dict:
            doc_dict[k] = doc_dict[k] / length

        # add the normalized document to the corpus
        self.documents.append([doc_name, doc_dict])

    def similarities(self, list_of_words):
        """Returns a list of all the [docname, similarity_score] pairs relative to a list of words.

        """

        # building the query dictionary
        query_dict = {}
        for w in list_of_words:
            query_dict[w] = query_dict.get(w, 0.0) + 1.0

        # normalizing the query
        length = float(len(list_of_words))
        for k in query_dict:
            query_dict[k] = query_dict[k] / length

        # computing the list of similarities
        sims = []
        for doc in self.documents:
            score = 0.0
            doc_dict = doc[1]
            for k in query_dict:
                if k in doc_dict:
                    score += (query_dict[k] / self.corpus_dict[k]) + (
                      doc_dict[k] / self.corpus_dict[k])
            sims.append([doc[0], score])

        return sims

    def get_top_k_important_words(self, top_k=10):
        words_list, tfidt_scores_list = list(), list()
        for document in self.documents:
            tfidf_scores_dict = document[1]
            for key in tfidf_scores_dict.keys():
                words_list.append(key)
                tfidt_scores_list.append(tfidf_scores_dict.get(key))
        uniq_words_list = list()
        for i in range(len(words_list)):
            if words_list[i] not in uniq_words_list:
                uniq_words_list.append(words_list[i])

        accum_tfidf_scores_list = [0.0 for _ in range(len(uniq_words_list))]
        for uniq_word_index in range(len(uniq_words_list)):
            for word_index in range(len(words_list)):
                if uniq_words_list[uniq_word_index] == words_list[word_index]:
                    accum_tfidf_scores_list[uniq_word_index] += tfidt_scores_list[word_index]

        # select top k words with the k largest accumulated tfidf scores
        representative_words, representative_tfidf_scores = list(), list()
        highest_tfidt_score = max(accum_tfidf_scores_list)
        most_important_keyword = uniq_words_list[accum_tfidf_scores_list.index(highest_tfidt_score)]
        for i in range(top_k):
            if len(accum_tfidf_scores_list) == 0:
                representative_words.append(most_important_keyword)
                representative_tfidf_scores.append(highest_tfidt_score)
                continue
            maximum = max(accum_tfidf_scores_list)
            representative_tfidf_scores.append(maximum)
            keyword = uniq_words_list[accum_tfidf_scores_list.index(maximum)]
            representative_words.append(keyword)
            accum_tfidf_scores_list.remove(maximum)
            uniq_words_list.remove(keyword)

        return representative_words, representative_tfidf_scores

if __name__ == '__main__':
    from representors.semantic_mts_representor import Semantic_MTS_Representor
    top_k_keywords_sizes = list(range(10, 51, 10))
    fixed_matrix_sizes = [Constants.FIXED_INPUT_MATRIX_SIZE for i in range(len(top_k_keywords_sizes))]
    vector_dimensionality_sizes = [int(fixed_matrix_sizes[i] / top_k_keywords_sizes[i]) for i in range(len(top_k_keywords_sizes))]

    mts_param_dict = {'etl_component': Constants.MY_ETL_COMPONENTS[3],
                      'top_k_keywords_sizes': top_k_keywords_sizes,
                      'vector_dimensionality_sizes': vector_dimensionality_sizes}
    my_generator = Semantic_MTS_Representor({}, 'Test-2', mts_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()
    print()
