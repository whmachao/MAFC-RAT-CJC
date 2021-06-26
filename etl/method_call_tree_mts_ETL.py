# Extract the multivariate time series from the method call tree
import numpy as np
import pandas as pd
from heapq import nlargest
from treelib import Node, Tree

from utilities.Utils import print_full_dataframe, get_normalized_list, draw_2D_figure, extract_words, load_words_dict


class Method_Call_Tree_MTS_Extractor:
    def __init__(self, method_call_tree, tree_level, n_gram):
        if not isinstance(method_call_tree, Tree):
            raise ValueError('Expect method_call_tree as a Tree, but got ' + str(type(method_call_tree)))
        self.method_call_tree = method_call_tree
        if tree_level is not None:
            raise ValueError('Expect tree_level as None, but got ' + str(type(tree_level)))
        self.tree_level = tree_level
        if n_gram is not None:
            raise ValueError('Expect n_gram as None, but got ' + str(n_gram))
        self.n_gram = n_gram
        self.method_call_sequence = self._transform_method_call_tree_to_call_sequence_preserving_list()
        print()

    def get_time_series_of_all_attributes(self):
        sample_vector_list = list()

        method_duration_ts = self._get_method_duration_ts()
        variable_read_ts = self._get_variable_access_times_ts(event=3)
        variable_write_ts = self._get_variable_access_times_ts(event=4)

        sample_vector_list.append(method_duration_ts)
        sample_vector_list.append(variable_read_ts)
        sample_vector_list.append(variable_write_ts)

        print('Start validation on sample_vector_list')
        for vector in sample_vector_list:
            if len(vector) == 0:
                raise ValueError('Any vector must NOT be an empty list!')
        print('Validation completed on sample_vector_list')

        return sample_vector_list

    # Extract the duration of each method as the sequence it is called
    def _get_method_duration_ts(self):
        method_duration_time_series = list()
        for node in self.method_call_sequence:
            my_df = node.data[0].reset_index(drop=True)
            duration = my_df.loc[len(my_df)-1]['threadClock'] - my_df.loc[0]['threadClock']
            method_duration_time_series.append(duration)
        if Constants.NORMALIZE_TIME_SERIES:
            method_duration_time_series = get_normalized_list(method_duration_time_series)
        return method_duration_time_series

    # Extract the variable access times (excluding those in all callees) of each method as the sequence it is accessed
    def _get_variable_access_times_ts(self, event=None):
        variable_access_times_time_series = []
        for node in self.method_call_sequence:
            access_times = self._get_variable_access_times_by_method(node, event=event)
            variable_access_times_time_series.append(access_times)
        if Constants.NORMALIZE_TIME_SERIES:
            variable_access_times_time_series = get_normalized_list(variable_access_times_time_series)
        return variable_access_times_time_series

    def _get_variable_access_times_by_method(self, node, event):
        my_df = node.data[0].reset_index(drop=True)
        method_name = my_df.loc[0]['method']
        if event is not None:
            if event not in [3, 4]:
                raise ValueError('Expect event to be 3 or 4, but got ' + str(event))
            variable_access_df = my_df[my_df['event'] == event]
        else:
            variable_access_df = my_df[my_df['event'].isin([3, 4])]
        access_times = 0
        for index, row in variable_access_df.iterrows():
            if row['method'] == method_name:
                access_times += 1
        return access_times

    # Use depth-first search to transform the method call tree into a call time preserving list
    # We re-order the time sequence among methods, which must follow the start clock ascending order
    def _transform_method_call_tree_to_call_sequence_preserving_list(self):
        # self.method_call_tree.show()
        method_call_sequence = list()
        candidate_nodes = self.method_call_tree.all_nodes()[1:]   # Exclude the root from the method call sequence
        for node in candidate_nodes:
            my_threadone_df = node.data[0]
            my_threadone_df = my_threadone_df.reset_index(drop=True)
            method, clock = my_threadone_df.loc[0]['method'], my_threadone_df.loc[0]['threadClock']
            if len(method_call_sequence) == 0:
                method_call_sequence.append(node)
                continue

            if len(method_call_sequence) == 1:
                temp_df = method_call_sequence[0].data[0].reset_index(drop=True)
                latest_clock = temp_df.loc[0]['threadClock']
                if clock > latest_clock:
                    method_call_sequence.append(node)
                else:
                    method_call_sequence.insert(0, node)
                continue

            for i in range(len(method_call_sequence) - 1, 0, -1):
                temp_df = method_call_sequence[i].data[0].reset_index(drop=True)
                latest_clock = temp_df.loc[0]['threadClock']
                if clock > latest_clock:
                    method_call_sequence.insert(i + 1, node)
                    break
                if i == 0:
                    method_call_sequence.insert(0, node)
                    break
        # if you do not want to filter the method call sequence, you simply return 'method_call_sequence'
        # return method_call_sequence
        return self._filter_method_call_sequence(method_call_sequence, Constants.DURATION_PERCENTAGE,
                                                 Constants.VARIABLE_ACCESS_TIMES, Constants.TOP_K_DURATION)

    # Filter out the important method calls according to the rules as follows:
    # Rule 1: method calls of which the duration is below 0.1% (0.001) of the whole duration;
    # Rule 2: method calls which do not access(read/write) any variable during the execution;
    def _filter_method_call_sequence(self, method_call_sequence, duration_percentage, variable_access_times, top_k_duration):
        if not isinstance(duration_percentage, float) or duration_percentage < 0.00001 or duration_percentage > 0.99999:
            raise ValueError('Expect duration_percentage as a float in (0, 1), but got ' + str(duration_percentage))
        if not isinstance(variable_access_times, int) or variable_access_times < 1:
            raise ValueError('Expect variable_access_times as an integer greater than 1, but got ' + str(variable_access_times))
        if not isinstance(top_k_duration, int) or top_k_duration < 3:
            raise ValueError('Expect top_k_duration as an integer greater than 2, but got ' + str(top_k_duration))
        filtered_method_call_sequence = list()
        root = self.method_call_tree.get_node(self.method_call_tree.root)
        trace_threadone_df = root.data[0].reset_index(drop=True)
        last_row_index = len(trace_threadone_df) - 1
        total_duration = trace_threadone_df.loc[last_row_index]['threadClock'] - trace_threadone_df.loc[0]['threadClock']
        print('Total duration: ' + str(total_duration))
        threshhold_duration = int(total_duration * duration_percentage)
        rule_one_filtered_num, rule_two_filtered_num, rule_three_filtered_num = 0, 0, 0
        for node in method_call_sequence:
            my_threadone_df = node.data[0].reset_index(drop=True)
            last_row_index = len(my_threadone_df) - 1
            my_duration = my_threadone_df.loc[last_row_index]['threadClock'] - my_threadone_df.loc[0]['threadClock']
            print('Current method duration: ' + str(my_duration))
            if my_duration < threshhold_duration:
                rule_one_filtered_num += 1
                continue
            my_variable_access_times = self._get_variable_access_times_by_method(node, event=None)
            if my_variable_access_times < variable_access_times:
                rule_two_filtered_num += 1
                continue
            filtered_method_call_sequence.append(node)
        duration_list = list()
        for node in filtered_method_call_sequence:
            my_threadone_df = node.data[0].reset_index(drop=True)
            last_row_index = len(my_threadone_df) - 1
            my_duration = my_threadone_df.loc[last_row_index]['threadClock'] - my_threadone_df.loc[0]['threadClock']
            duration_list.append(my_duration)
        my_k_top_duration_list = nlargest(top_k_duration, duration_list)
        new_filtered_method_call_sequence = list()
        for node in filtered_method_call_sequence:
            my_threadone_df = node.data[0].reset_index(drop=True)
            last_row_index = len(my_threadone_df) - 1
            my_duration = my_threadone_df.loc[last_row_index]['threadClock'] - my_threadone_df.loc[0]['threadClock']
            if my_duration in my_k_top_duration_list:
                new_filtered_method_call_sequence.append(node)
                continue
            rule_three_filtered_num += 1
        total_method_calls = len(self.method_call_tree.all_nodes())
        print('Total method calls: ' + str(total_method_calls))
        print('rule_one_filtered_num: ' + str(rule_one_filtered_num))
        print('rule_two_filtered_num: ' + str(rule_two_filtered_num))
        print('rule_three_filtered_num: ' + str(rule_three_filtered_num))
        left_method_calls = total_method_calls - rule_one_filtered_num - rule_two_filtered_num - rule_three_filtered_num
        reserved_percentage = np.round(left_method_calls/total_method_calls, 3)
        print('Method calls left: ' + str(left_method_calls) + ', ' + str(reserved_percentage) + ' reserved')

        return new_filtered_method_call_sequence


if __name__ == '__main__':
    import utilities.Constants as Constants
    from representors.mts_representor import MTS_Representor

    mts_param_dict = {'etl_component': Constants.MY_ETL_COMPONENTS[2]}
    my_generator = MTS_Representor({}, 'Test-2', mts_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()
    print()
