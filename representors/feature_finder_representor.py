import numpy as np
import time
import os
import utilities.Constants as Constants
from utilities.Utils import construct_method_call_tree, create_etl_component, get_project_dir
from utilities.Utils import simple_distribution_transform, pad_method_call_tree, split_train_test
import sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class Feature_Finder_Representor():
    def __init__(self, datasets_dict, dataset_name, param_dict):
        self.datasets_dict = datasets_dict
        self.dataset_name = dataset_name
        self.param_dict = param_dict

    def get_all_representations_dict(self):
        top_k_keywords_sizes = self.param_dict['top_k_keywords_sizes']
        app_trace_dict = self._get_app_trace_dict()
        all_representation_dict = {}
        total_representations = len(top_k_keywords_sizes)*len(Constants.SVM_C_VALUES)*len(Constants.KNN_K_VALUES)*len(Constants.KNN_STRATEGY_LIST)*len(Constants.KNN_DISTANCE_TYPES)
        progress = 0

        for i in range(len(top_k_keywords_sizes)):
            for c in range(len(Constants.SVM_C_VALUES)):
                for j in range(len(Constants.KNN_K_VALUES)):
                    for k in range(len(Constants.KNN_STRATEGY_LIST)):
                        for r in range(len(Constants.KNN_DISTANCE_TYPES)):
                            progress += 1
                            top_k_keywords = top_k_keywords_sizes[i]
                            svm_c = Constants.SVM_C_VALUES[c]
                            knn_k = Constants.KNN_K_VALUES[j]
                            knn_strategy = Constants.KNN_STRATEGY_LIST[k]
                            knn_distance = Constants.KNN_DISTANCE_TYPES[r]
                            representation_key = str(top_k_keywords) + '_' + str(svm_c) + '_' + str(knn_k) + '_' \
                                                 + str(knn_strategy) + '_' + knn_distance
                            start_time = time.time()
                            print('##################################################################################')
                            print('Start to generate ' + representation_key + ': ' + str(progress) + ' out of ' + str(
                                total_representations))
                            my_representation_dict = self._get_fixed_length_representation(app_trace_dict,
                                                                                           top_k_keywords)
                            all_representation_dict[representation_key] = my_representation_dict[self.dataset_name]
                            complete_time = time.time()
                            consumed_time = complete_time - start_time
                            print(
                                'Complete to generate ' + representation_key + ': ' + str(progress) + ' out of ' + str(
                                    total_representations) + ' (' + str(consumed_time) + ' seconds consumed)')
                            print('##################################################################################')

        return all_representation_dict

    def _get_app_trace_dict(self):
        # Step 1: extract the method call tree from the original trace file
        print('Start to parse: ' + self.dataset_name)
        start_time = time.time()
        app_trace_dict = {}
        mct_depth_list, mct_nodes_num_list = list(), list()
        all_category_dir = get_project_dir() + '/archives' + '/' + Constants.ARCHIVE_NAMES[0]
        category_dir = all_category_dir + '/' + self.dataset_name
        for app_name in os.listdir(category_dir):
            trace_names_list = os.listdir(category_dir + '/' + app_name)
            for trace_name in trace_names_list:
                my_trace_url = category_dir + '/' + app_name + '/' + trace_name
                my_trace_name = self.dataset_name + '&' + app_name + '&' + trace_name
                call_method_tree, exceptional_lines = construct_method_call_tree(my_trace_name, my_trace_url)
                app_trace_dict[my_trace_name] = call_method_tree
                if Constants.SHOW_DETAILS:
                    call_method_tree.show()
                print(my_trace_name + ' method call tree statistics:')
                print('Depth: ' + str(call_method_tree.depth()))
                print('No. of Nodes: ' + str(len(call_method_tree.all_nodes())))
                print('Lines not covered: ' + str(exceptional_lines))
                mct_depth_list.append(call_method_tree.depth())
                mct_nodes_num_list.append(len(call_method_tree.all_nodes()))
                print('---------------------------------------------------------------------------------')
        complete_time = time.time()
        print('Complete to parse: ' + self.dataset_name + ' using ' + str(complete_time - start_time) + ' seconds!')
        if Constants.DRAW_STATISTICAL_CHARACTERISTICS:
            my_title = self.dataset_name
            my_data_dict = dict()
            my_data_dict['x_label'], my_data_dict['y_label'] = 'MCT Depth', 'MCT Nodes'
            my_data_dict['x_values'], my_data_dict['y_values'] = mct_depth_list, mct_nodes_num_list

        return app_trace_dict

    # Multivariate time series: start clock of each method as the unified a-aix ticks
    # TS1: Method Duration, the execution time of each method call;
    # TS2: Variable Read, number of variable readings during the execution of each method call;
    # TS3: Variable Write, number of variable writings during the execution of each method call;
    def _get_fixed_length_representation(self, app_trace_dict, k_keywords):
        # Extract the time series values of each attribute for the specific tree level from the method call tree
        true_labels_list, split_labels_list = [], []
        all_attributes_of_all_traces = []
        for key in app_trace_dict.keys():
            true_label = (key.split('&')[-1]).split('-')[0]
            true_labels_list.append(true_label)
            split_label = key.split('&')[1]
            split_labels_list.append(split_label)
            call_method_tree = app_trace_dict[key]
            # my_nodes = get_nodes_at_specific_level(call_method_tree, tree_level)
            print('Attribute Extraction of ' + str(key) + ' started ... ...')
            etl_param_dict = {'k_keywords': k_keywords}
            my_etl_component = create_etl_component(self.param_dict['etl_component'], call_method_tree, etl_param_dict)
            sample_vector_list = my_etl_component.get_time_series_of_all_attributes()
            print('Attribute Extraction of ' + str(key) + ' has been completed!')
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            all_attributes_of_all_traces.append(sample_vector_list)

        for trace_attributes_list in all_attributes_of_all_traces:
            for attribute_list in trace_attributes_list:
                if len(attribute_list) == 0:
                    raise ValueError('Any attribute must not be an empty list!')
        print('All attributes are non-empty list!')

        # Step 2: obtain the proper representation for each keyword extracted from the execution traces
        # Solution 1: use word2vec to transform the keywords into vector format while preserving semantic similarity
        num_of_attributes = len(all_attributes_of_all_traces[0])
        tfidf_arrays_in_lists_by_attribute = []
        for attribute_index in range(num_of_attributes):
            trace_keywords_list_in_lists = []
            for trace_attributes_list in all_attributes_of_all_traces:
                trace_keywords_list_in_lists.append(trace_attributes_list[attribute_index])
            corpus = self._get_corpus_from_keywords(trace_keywords_list_in_lists)
            keywords_tfidf_arrays = self._compute_tfidf_for_keywords(corpus)
            tfidf_arrays_in_lists_by_attribute.append(keywords_tfidf_arrays)
        print('Total number of histogram based attributes: ' + str(len(tfidf_arrays_in_lists_by_attribute)))
        print('Total number of histogram based samples(traces): ' + str(len(tfidf_arrays_in_lists_by_attribute[0])))

        # Step 3: split all samples into the train/test sets
        train_samples_list, train_labels_list, test_samples_list, test_labels_list = split_train_test(true_labels_list,
                                                                                                      split_labels_list,
                                                                                                      num_of_attributes,
                                                                                                      tfidf_arrays_in_lists_by_attribute)
        if train_labels_list is None:
            raise ValueError('Train Test Split failed!')

        # Step 4: change labels' format from 'String' to 'Integer' starting with 0
        unique_labels_list = list(set(true_labels_list))
        for train_label_index in range(len(train_labels_list)):
            integer_label = unique_labels_list.index(train_labels_list[train_label_index])
            train_labels_list[train_label_index] = integer_label
        for test_label_index in range(len(test_labels_list)):
            integer_label = unique_labels_list.index(test_labels_list[test_label_index])
            test_labels_list[test_label_index] = integer_label

        train_samples_list = np.array(train_samples_list)
        train_labels_list = np.array(train_labels_list)
        test_samples_list = np.array(test_samples_list)
        test_labels_list = np.array(test_labels_list)

        my_raw_representation_dict = {self.dataset_name: (train_samples_list.copy(), train_labels_list.copy(),
                                                          test_samples_list.copy(), test_labels_list.copy())}

        print('No. of Classes: ' + str(len(unique_labels_list)))
        print('No. of Attributes: ' + str(num_of_attributes))
        print('Train/Test Size: ' + str(train_samples_list.shape[0]) + '/' + str(test_samples_list.shape[0]))
        print('Time Series Length: ' + str(train_samples_list.shape[1]))

        return my_raw_representation_dict

    # Organize the corpus from keywords.
    def _get_corpus_from_keywords(self, keywords_as_multipl_lists):
        corpus = list()
        for keywords_list in keywords_as_multipl_lists:
            document = ''
            for keyword in keywords_list:
                if document == '':
                    document = document + keyword
                    continue
                document = document + ' ' + keyword
            corpus.append(document)

        return corpus

    # Compute the tf and tf-idf scores of each keyword occurred in the corpus.
    def _compute_tfidf_for_keywords(self, corpus):
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        keywords = vectorizer.get_feature_names()
        doc_tfidf_vectors = tfidf.toarray()

        return doc_tfidf_vectors

if __name__ == '__main__':
    top_k_keywords_sizes = list(range(10, 21, 10))  # No. of keywords extracted from each execution trace.
    vector_dimensionality_sizes = [2]  # Each keyword's representation contains 2 values: tf and tf-idf.

    mts_param_dict = {'etl_component': Constants.MY_ETL_COMPONENTS[3],
                      'top_k_keywords_sizes': top_k_keywords_sizes,
                      'vector_dimensionality_sizes': vector_dimensionality_sizes}
    my_generator = Feature_Finder_Representor({}, 'Test', mts_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()
    print()
