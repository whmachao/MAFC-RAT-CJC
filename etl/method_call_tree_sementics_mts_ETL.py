# Extract the "semantic" multivariate time series from the method call tree
import numpy as np
import pandas as pd
from heapq import nlargest
from treelib import Node, Tree

from utilities.Utils import get_project_dir, get_normalized_list, draw_2D_figure, extract_words, load_words_dict
from utilities.tfidf import TfIdf
import utilities.Constants as Constants


class Method_Call_Tree_Semantic_MTS_Extractor:
    def __init__(self, method_call_tree, top_k, not_used):
        if not isinstance(method_call_tree, Tree):
            raise ValueError('Expect method_call_tree as a Tree, but got ' + str(type(method_call_tree)))
        self.method_call_tree = method_call_tree
        self.top_k = top_k
        self.method_call_sequence = self._transform_method_call_tree_to_call_sequence_preserving_list()
        print()

    def get_time_series_of_all_attributes(self):
        sample_vector_list = list()

        my_words = load_words_dict(get_project_dir() + '/resources/words.txt')
        method_full_name_ts = self._get_method_full_name_ts()
        method_full_name_ts = self._transfer_method_name_sequence_into_words_sequence(method_full_name_ts, my_words)
        variable_read_full_name_ts = self._get_variable_full_name_ts(event=3)
        variable_read_full_name_ts = self._transfer_variable_name_sequence_into_words_sequence(variable_read_full_name_ts, my_words)
        variable_write_full_name_ts = self._get_variable_full_name_ts(event=4)
        variable_write_full_name_ts = self._transfer_variable_name_sequence_into_words_sequence(variable_write_full_name_ts, my_words)

        table = TfIdf()
        table.add_document('method-call', self._transfer_lists_in_list_into_unnested_list(method_full_name_ts))
        table.add_document('variable-read', self._transfer_lists_in_list_into_unnested_list(variable_read_full_name_ts))
        table.add_document('variable-write', self._transfer_lists_in_list_into_unnested_list(variable_write_full_name_ts))
        top_k_keywords, top_k_tfidf_scores = table.get_top_k_important_words(top_k=self.top_k)

        if Constants.RECORD_KEYWORDS_AND_TFIDF:
            results_df = pd.read_csv('keywords.csv')
            for index in range(len(top_k_keywords)):
                trace_name = self.method_call_tree.root
                dimensionality = self.top_k
                keyword = top_k_keywords[index]
                tfidf = top_k_tfidf_scores[index]
                new_result_df = pd.DataFrame([[trace_name, dimensionality, keyword, tfidf]], columns=results_df.columns)
                results_df = pd.concat((results_df, new_result_df), axis=0, sort=False)
            results_df.to_csv('keywords.csv', index=False)

        sample_vector_list.append(top_k_keywords)

        # sample_vector_list.append(method_full_name_ts)
        # sample_vector_list.append(variable_read_full_name_ts)
        # sample_vector_list.append(variable_write_full_name_ts)

        print('Start validation on sample_vector_list')
        for vector in sample_vector_list:
            if len(vector) == 0:
                raise ValueError('Any vector must NOT be an empty list!')
        print('Validation completed on sample_vector_list')

        return sample_vector_list

    # Extract the full name of each method as the sequence it is called
    def _get_method_full_name_ts(self):
        method_full_name_time_series = list()
        for node in self.method_call_sequence:
            method_full_name = self._get_method_full_name_by_method(node)
            method_full_name_time_series.append(method_full_name)
        return method_full_name_time_series

    def _get_method_full_name_by_method(self, node):
        my_df = node.data[0].reset_index(drop=True)
        method_identifier = my_df.loc[0]['method']
        my_event8_df = node.data[1].reset_index(drop=True)
        my_event8_df = my_event8_df[my_event8_df['retVal'] == method_identifier].reset_index(drop=True)
        method_full_name = my_event8_df.loc[0]['method']

        return method_full_name

    # Extract the full name of each variable as the sequence it is accessed (read, write, read/write)
    def _get_variable_full_name_ts(self, event=None):
        variable_full_name_time_series = list()
        for node in self.method_call_sequence:
            my_df = node.data[0].reset_index(drop=True)
            method_name = my_df.loc[0]['method']
            my_event9_df = node.data[2].reset_index(drop=True)
            # extract the variable access dataframe according to the event type (3: read, 5: write)
            if event is not None:
                if event not in [3, 4]:
                    raise ValueError('Expect event to be 3 or 4, but got ' + str(event))
                variable_df = my_df[my_df['event'] == event]
            else:
                variable_df = my_df[my_df['event'].isin([3, 4])]
            # look up the variable string name according to its memory address
            for index, row in variable_df.iterrows():
                if row['method'] == method_name:
                    temp_df = my_event9_df[my_event9_df['retVal'] == row['field']].reset_index(drop=True)
                    variable_full_name = temp_df.loc[0]['field']
                    variable_full_name_time_series.append(variable_full_name)
        return variable_full_name_time_series

    # Use depth-first search to transform the method call tree into a call sequence preserving list
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
    #
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

    # Transfer method full name sequence into discrete words sequence
    def _transfer_method_name_sequence_into_words_sequence(self, my_method_full_name_ts, my_words):
        words_sequence_list = list()
        for method_full_name in my_method_full_name_ts:
            method_short_name = method_full_name.split()[1].split('(')[0]
            words_list = extract_words(my_words, method_short_name)
            words_sequence_list.append(words_list)
        return words_sequence_list

    # Transfer variable name sequence into discrete words sequence
    def _transfer_variable_name_sequence_into_words_sequence(self, my_variable_full_name_ts, my_words):
        words_sequence_list = list()
        for variable_full_name in my_variable_full_name_ts:
            variable_short_name = variable_full_name.split()[0]
            words_list = extract_words(my_words, variable_short_name)
            words_sequence_list.append(words_list)
        return words_sequence_list

    # Transfer lists in list into a unnested list
    def _transfer_lists_in_list_into_unnested_list(self, nested_list):
        unnested_list = list()
        for sublist in nested_list:
            for item in sublist:
                unnested_list.append(item)
        return unnested_list


if __name__ == '__main__':
    # create a csv file to record the keywords and their tf-idf scores
    if Constants.RECORD_KEYWORDS_AND_TFIDF:
        res = pd.DataFrame(data=np.zeros((0, 4), dtype=np.float), index=[],
                           columns=['trace', 'dimensionality', 'keyword', 'tfidf'])
        res.to_csv('keywords.csv', index=False)

    from representors.semantic_mts_representor import Semantic_MTS_Representor
    # top_k_keywords_sizes = list(range(10, 51, 10))
    top_k_keywords_sizes = Constants.TOP_K_KEYWORDS_LIST
    fixed_matrix_sizes = [Constants.FIXED_INPUT_MATRIX_SIZE for i in range(len(top_k_keywords_sizes))]
    vector_dimensionality_sizes = [int(fixed_matrix_sizes[i] / top_k_keywords_sizes[i]) for i in range(len(top_k_keywords_sizes))]

    mts_param_dict = {'etl_component': Constants.MY_ETL_COMPONENTS[3],
                      'top_k_keywords_sizes': top_k_keywords_sizes,
                      'vector_dimensionality_sizes': vector_dimensionality_sizes}
    my_generator = Semantic_MTS_Representor({}, 'Test-2', mts_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()
    print()
