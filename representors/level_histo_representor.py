import numpy as np
import pandas as pd
import networkx as nx
from openpyxl import Workbook
import json
import time
import os
from operator import sub
import matplotlib.pyplot as plt
import utilities.Constants as Constants
from utilities.Utils import construct_method_call_tree, create_etl_component, simple_histogram_transformation
from utilities.Utils import simple_distribution_transform, pad_method_call_tree, split_train_test
import Checker
from treelib import Node, Tree
import ast
from utilities.Utils import get_minimum_from_lists, get_maximum_from_lists
import sys


class Level_Histo_Representor():
    def __init__(self, datasets_dict, dataset_name, param_dict):
        self.datasets_dict = datasets_dict
        self.dataset_name = dataset_name
        self.param_dict = param_dict
        self.deepest_tree_level = max(param_dict['tree_level_list'])

        if self.param_dict['tree_level_list'] is None:
            raise ValueError('Expect tree_level_list but got None!')

    # All-in-One representation: from level 1 to Level tree_level containing the method behavior and the field behavior
    # The method behavior: from 1-gram to n-gram call sequence;
    # The field behavior: from 1-gram to n-gram access sequence;
    def _get_fixed_length_representation(self, app_trace_dict, fixed_length, tree_level, n_gram):
        if not isinstance(self.dataset_name, str):
            raise ValueError('target_category must be of the type str!')
        if fixed_length < 3:
            raise ValueError('fixed_length must be an integer greater than 2!')

        # Step 2: extract the time series values of each attribute for the specific tree level from the method call tree
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
            my_etl_component = create_etl_component(self.param_dict['etl_component'], call_method_tree, tree_level, n_gram)
            sample_vector_list = my_etl_component.get_time_series_of_all_attributes()
            print('Attribute Extraction of ' + str(key) + ' has been completed!')
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            all_attributes_of_all_traces.append(sample_vector_list)

        for trace_attributes_list in all_attributes_of_all_traces:
            for attribute_list in trace_attributes_list:
                if len(attribute_list) == 0:
                    raise ValueError('Any attribute must not be an empty list!')
        print('All attributes are non-empty list!')

        num_of_attributes = len(all_attributes_of_all_traces[0])
        histogram_arrays_in_lists_by_attribute = []
        for attribute_index in range(num_of_attributes):
            trace_attributes_list_in_lists = []
            for trace_attributes_list in all_attributes_of_all_traces:
                trace_attributes_list_in_lists.append(trace_attributes_list[attribute_index])
            # Using simple histogram transform makes the representation contain too many zeros
            # new_attribute_histogram_arrays = simple_histogram_transformation(trace_attributes_list_in_lists, fixed_length)
            new_attribute_histogram_arrays = simple_distribution_transform(trace_attributes_list_in_lists)
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

    def _get_app_trace_dict(self):
        # Step 1: extract the method call tree from the original trace file
        print('Start to parse: ' + self.dataset_name)
        start_time = time.time()
        functionality_names_dict, app_trace_dict = {}, {}
        all_category_dir = Constants.ROOT_DIR + 'archives' + '/' + Constants.ARCHIVE_NAMES[2]
        category_dir = all_category_dir + '/' + self.dataset_name + '/'
        functionality_names_dict[self.dataset_name] = os.listdir(category_dir)
        for app_name in functionality_names_dict[self.dataset_name]:
            trace_names_list = os.listdir(category_dir + '/' + app_name)
            for trace_name in trace_names_list:
                my_trace_url = category_dir + '/' + app_name + '/' + trace_name
                my_trace_name = self.dataset_name + '&' + app_name + '&' + trace_name
                call_method_tree, exceptional_lines = construct_method_call_tree(my_trace_name, my_trace_url)
                app_trace_dict[my_trace_name] = call_method_tree
                # call_method_tree.show()
                print(my_trace_name + ' method call tree statistics:')
                print('Depth: ' + str(call_method_tree.depth()))
                print('No. of Nodes: ' + str(len(call_method_tree.all_nodes())))
                print('Lines not covered: ' + str(exceptional_lines))
                print('---------------------------------------------------------------------------------')
                # extend the method call tree to the tree_level
                if self.deepest_tree_level is not None:
                    padded_tree = pad_method_call_tree(call_method_tree, self.deepest_tree_level)
                    app_trace_dict[my_trace_name] = padded_tree
                    # padded_tree.show()
                    print(my_trace_name + ' PADDED method call tree statistics:')
                    print('Depth: ' + str(padded_tree.depth()))
                    print('No. of Nodes: ' + str(len(padded_tree.all_nodes())))
                print('*********************************************************************************')
        complete_time = time.time()
        print('Complete to parse: ' + self.dataset_name + ' using ' + str(complete_time - start_time) + ' seconds!')
        return app_trace_dict

    def get_all_representations_dict(self):
        fixed_length_list = self.param_dict['fixed_length_list']
        tree_level_list = self.param_dict['tree_level_list']
        gram_num_list = self.param_dict['gram_num_list']
        app_trace_dict = self._get_app_trace_dict()
        all_representation_dict, aux_info_dict = {}, {}
        total_representations = len(fixed_length_list) * len(tree_level_list) * len(gram_num_list)
        progress = 0
        for fixed_length in fixed_length_list:
            for tree_level in tree_level_list:
                for n_gram in gram_num_list:
                    progress += 1
                    representation_key = str(fixed_length) + '_1-' + str(tree_level) + '_1-' + str(n_gram)
                    start_time = time.time()
                    print('##################################################################################')
                    print('Start to generate '+representation_key+': '+str(progress)+' out of '+str(total_representations))
                    my_representation_dict = self._get_fixed_length_representation(app_trace_dict, fixed_length, tree_level, n_gram)
                    my_current_representation = my_representation_dict[self.dataset_name]
                    all_representation_dict[representation_key] = my_current_representation
                    complete_time = time.time()
                    consumed_time = complete_time - start_time
                    mem_size_in_bytes = sys.getsizeof(my_current_representation)
                    aux_info_dict[representation_key] = [round(consumed_time, 3), mem_size_in_bytes]
                    print('Complete to generate '+representation_key+': '+str(progress)+' out of '+str(total_representations)+' ('+str(consumed_time)+' seconds consumed)')
                    print('##################################################################################')

        return all_representation_dict, aux_info_dict


if __name__ == '__main__':
    level_histo_param_dict = {'tree_level_list': Constants.TREE_LEVEL_LIST,
                              'fixed_length_list': Constants.FIXED_LENGTH_LIST,
                              'gram_num_list': Constants.GRAM_NUMBER_LIST,
                              'etl_component': Constants.MY_ETL_COMPONENTS[1]}
    my_generator = Level_Histo_Representor({}, 'Test', level_histo_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()
    print()