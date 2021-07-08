import numpy as np
import os
import utilities.Constants as Constants
from utilities.Utils import get_project_dir, simple_histogram_transformation, create_etl_component
from utilities.Utils import get_train_test_indices, construct_method_call_tree, get_nodes_at_specific_level


class Plain_Histo_Representor():
    def __init__(self, datasets_dict, dataset_name, param_dict):
        self.datasets_dict = datasets_dict
        self.dataset_name = dataset_name
        self.param_dict = param_dict

    def _get_histograms_based_representation(self, bin_number):
        if not isinstance(self.dataset_name, str):
            raise ValueError('target_category must be of the type str!')
        if bin_number < 3:
            raise ValueError('bin_number must be an integer greater than 2!')

        # Step 1: extract the method call tree from the original trace file
        functionality_names_dict, app_trace_dict = {}, {}
        all_category_dir = get_project_dir() + '/archives' + '/' + Constants.ARCHIVE_NAMES[0]
        category_dir = all_category_dir + '/' + self.dataset_name
        functionality_names_dict[self.dataset_name] = os.listdir(category_dir)
        for app_name in functionality_names_dict[self.dataset_name]:
            trace_names_list = os.listdir(category_dir + '/' + app_name)
            for trace_name in trace_names_list:
                my_trace_url = category_dir + '/' + app_name + '/' + trace_name
                my_trace_name = self.dataset_name + '&' + app_name + '&' + trace_name
                call_method_tree, exceptional_lines = construct_method_call_tree(my_trace_name, my_trace_url)
                app_trace_dict[my_trace_name] = call_method_tree
                print(my_trace_name + ' method call tree statistics:')
                print('Depth: ' + str(call_method_tree.depth()))
                print('No. of Nodes: ' + str(len(call_method_tree.all_nodes())))
                print('Lines not covered: ' + str(exceptional_lines))
                print('*************************************************************************************')

        # Step 2: extract the time series values for each attribute from the given temp_df
        labels_list = []
        all_attributes_of_all_traces = []
        for key in app_trace_dict.keys():
            sample_label = (key.split('&')[1]).split('-')[0]
            labels_list.append(sample_label)
            call_method_tree = app_trace_dict[key]
            my_nodes = get_nodes_at_specific_level(call_method_tree, 0)  # only take the root of the method call tree
            my_etl_component = create_etl_component(self.param_dict['etl_component'], my_nodes, dict())
            sample_vector_list = my_etl_component.get_time_series_of_all_attributes()
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
            new_attribute_histogram_arrays = simple_histogram_transformation(trace_attributes_list_in_lists, bin_number)
            histogram_arrays_in_lists_by_attribute.append(new_attribute_histogram_arrays)
        print('Total number of histogram based attributes: ' + str(len(histogram_arrays_in_lists_by_attribute)))
        print('Total number of histogram based samples(traces): ' + str(len(histogram_arrays_in_lists_by_attribute[0])))

        # Step 3: split all samples into the train/test sets
        train_test_split_ratio = Constants.TRAIN_TEST_SPLIT_RATIO
        # First, step1, obtain indices for train and test sets
        train_indices, test_indices = get_train_test_indices(train_test_split_ratio, labels_list)
        train_samples_list, train_labels_list, test_samples_list, test_labels_list = [], [], [], []
        for train_index in train_indices:
            train_sample = []
            for attribute_index in range(num_of_attributes):
                train_sample = train_sample + list(histogram_arrays_in_lists_by_attribute[attribute_index][train_index])
            train_samples_list.append(train_sample)
            train_labels_list.append(labels_list[train_index])
        for test_index in test_indices:
            test_sample = []
            for attribute_index in range(num_of_attributes):
                test_sample = test_sample + list(histogram_arrays_in_lists_by_attribute[attribute_index][test_index])
            test_samples_list.append(test_sample)
            test_labels_list.append(labels_list[test_index])

        # Step 4: change labels' format from 'String' to 'Integer' starting with 0
        unique_labels_list = list(set(labels_list))
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

    def get_all_representations_dict(self):
        bin_num_list = self.param_dict['bin_num_list']
        all_representation_dict = {}
        tree_level = 0
        for bin_number in bin_num_list:
            my_representation_dict = self._get_histograms_based_representation(bin_number)
            all_representation_dict[str(bin_number)+'_'+str(tree_level)] = my_representation_dict[self.dataset_name]

        return all_representation_dict


if __name__ == '__main__':
    plain_histo_param_dict = {'tree_level_list': None,
                              'bin_num_list': Constants.BIN_NUMBER_LIST,
                              'etl_component': Constants.MY_ETL_COMPONENTS[0]}
    my_generator = Plain_Histo_Representor({}, 'Test', plain_histo_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()
    print()