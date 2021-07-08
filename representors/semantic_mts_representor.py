import numpy as np
import time
import os
import utilities.Constants as Constants
from utilities.Utils import construct_method_call_tree, create_etl_component, get_project_dir
from utilities.Utils import simple_distribution_transform, split_train_test, ecdf_based_transform, draw_2D_figure
from gensim.models import Word2Vec


class Semantic_MTS_Representor():
    def __init__(self, datasets_dict, dataset_name, param_dict):
        self.datasets_dict = datasets_dict
        self.dataset_name = dataset_name
        self.param_dict = param_dict

    def get_all_representations_dict(self):
        top_k_keywords_sizes = self.param_dict['top_k_keywords_sizes']
        vector_dimensionality_sizes = self.param_dict['vector_dimensionality_sizes']
        app_trace_dict = self._get_app_trace_dict()
        all_representation_dict = {}
        total_representations = len(top_k_keywords_sizes) * len(vector_dimensionality_sizes)
        progress = 0
        for i in range(len(top_k_keywords_sizes)):
            progress += 1
            top_k_keywords, vector_dimension = top_k_keywords_sizes[i], vector_dimensionality_sizes[i]
            representation_key = str(top_k_keywords) + '_' + str(vector_dimension)
            start_time = time.time()
            print('##################################################################################')
            print('Start to generate ' + representation_key + ': ' + str(progress) + ' out of ' + str(total_representations))
            my_representation_dict = self._get_fixed_length_representation(app_trace_dict, top_k_keywords, vector_dimension)
            all_representation_dict[representation_key] = my_representation_dict[self.dataset_name]
            complete_time = time.time()
            consumed_time = complete_time - start_time
            print('Complete to generate ' + representation_key + ': ' + str(progress) + ' out of ' + str(
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
            draw_2D_figure(my_title, my_data_dict)

        return app_trace_dict

    # Multivariate time series: start clock of each method as the unified a-aix ticks
    # TS1: Method Duration, the execution time of each method call;
    # TS2: Variable Read, number of variable readings during the execution of each method call;
    # TS3: Variable Write, number of variable writings during the execution of each method call;
    def _get_fixed_length_representation(self, app_trace_dict, k_keywords, vector_dimension):
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
            if Constants.DRAW_STATISTICAL_CHARACTERISTICS:
                for vector in sample_vector_list:
                    my_title = key + '-Attribute-' + str(sample_vector_list.index(vector))
                    my_data_dict = dict()
                    my_data_dict['x_label'], my_data_dict['y_label'] = 'Call Sequence', 'Value'
                    my_data_dict['x_values'], my_data_dict['y_values'] = range(len(vector)), vector
                    draw_2D_figure(my_title, my_data_dict, legend=None, y_log_scale=False, pic_size=(6, 3), pic_type='bars')
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            all_attributes_of_all_traces.append(sample_vector_list)

        for trace_attributes_list in all_attributes_of_all_traces:
            for attribute_list in trace_attributes_list:
                if len(attribute_list) == 0:
                    raise ValueError('Any attribute must not be an empty list!')
        print('All attributes are non-empty list!')

        # Step 2: key step is to obtain the proper representation for each key word extracted from the traces
        # Solution 1: use word2vec to transform the keywords into vector format while preserving semantic similarity
        num_of_attributes = len(all_attributes_of_all_traces[0])
        histogram_arrays_in_lists_by_attribute = []
        for attribute_index in range(num_of_attributes):
            trace_keywords_list_in_lists = []
            for trace_attributes_list in all_attributes_of_all_traces:
                trace_keywords_list_in_lists.append(trace_attributes_list[attribute_index])
            new_attribute_histogram_arrays = self._word_2_vector_transform(trace_keywords_list_in_lists, vector_dimension)
            histogram_arrays_in_lists_by_attribute.append(new_attribute_histogram_arrays)
        print('Total number of histogram based attributes: ' + str(len(histogram_arrays_in_lists_by_attribute)))
        print('Total number of histogram based samples(traces): ' + str(len(histogram_arrays_in_lists_by_attribute[0])))

        # Step 3: split all samples into the train/test sets
        train_samples_list, train_labels_list, test_samples_list, test_labels_list = split_train_test(true_labels_list,
                                                                                                      split_labels_list,
                                                                                                      num_of_attributes,
                                                                                                      histogram_arrays_in_lists_by_attribute)
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

    def _word_2_vector_transform(self, lists_in_list, vector_dimension):
        if not isinstance(lists_in_list, list):
            raise ValueError('lists_in_list must be the type of list!')
        for my_list in lists_in_list:
            if len(my_list) == 0:
                raise ValueError('All sub-lists must be non-empty!')

        model = Word2Vec(lists_in_list, size=vector_dimension, window=5, min_count=1, workers=4)
        model.save("word2vec.model")
        new_lists_in_list = []
        for my_list in lists_in_list:
            new_list = list()
            for word in my_list:
                vector = model.wv[word]
                new_list += list(vector)
                # print(word + ' word2vec vector: ' + str(vector))
            new_lists_in_list.append(new_list)

        return new_lists_in_list


if __name__ == '__main__':
    top_k_keywords_sizes = list(range(10, 51, 10))
    fixed_matrix_sizes = [Constants.FIXED_INPUT_MATRIX_SIZE for i in range(len(top_k_keywords_sizes))]
    vector_dimensionality_sizes = [int(fixed_matrix_sizes[i] / top_k_keywords_sizes[i]) for i in range(len(top_k_keywords_sizes))]

    mts_param_dict = {'etl_component': Constants.MY_ETL_COMPONENTS[3],
                      'top_k_keywords_sizes': top_k_keywords_sizes,
                      'vector_dimensionality_sizes': vector_dimensionality_sizes}
    my_generator = Semantic_MTS_Representor({}, 'Test', mts_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()
    print()