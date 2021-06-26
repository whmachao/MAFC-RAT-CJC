# TEST
import numpy as np
import pandas as pd
from treelib import Node, Tree

from utilities.Utils import new_process_single_trace, get_nodes_at_specific_level, print_full_dataframe


class Level_Histo_Extractor:
    def __init__(self, method_call_tree, tree_level, n_gram):
        if not isinstance(method_call_tree, Tree):
            raise ValueError('Expect method_call_tree as a Tree, but got ' + str(type(method_call_tree)))
        self.method_call_tree = method_call_tree
        if not isinstance(tree_level, int) and 0 < tree_level < method_call_tree.depth() + 1:
            raise ValueError('Expect tree_level as an Integer in range (0,'+str(method_call_tree.depth())+'), but got ' + str(type(tree_level)))
        self.tree_level = tree_level
        if not isinstance(n_gram, int) and n_gram > 0:
            raise ValueError('Expect n_gram as an integer greater than 1, but got ' + str(n_gram))
        self.n_gram = n_gram
        print()

    def _get_processed_df_at_specific_level(self, nodes_at_specific_level):
        if len(nodes_at_specific_level) < 1:
            raise ValueError('nodes_at_specific_level MUST have at least one element!')
        nodes = nodes_at_specific_level
        for node in nodes:
            if not isinstance(node, Node):
                raise ValueError('node MUST be the type of Node!')
            if not isinstance(node.data[0], pd.DataFrame) or not isinstance(node.data[1], pd.DataFrame) or not isinstance(node.data[2], pd.DataFrame):
                raise ValueError('node.data MUST be the type of Dataframe!')
        processed_df = pd.DataFrame()
        for node in nodes:
            processed_df = pd.concat([processed_df, new_process_single_trace(node.data[0], node.data[1])], axis=0, ignore_index=True)

        return processed_df

    def _get_attribute_n_gram_of_method_calls(self, processed_df, n_gram):
        # return [1] if n_gram is equal to or greater than the total number of method calls
        if n_gram >= len(processed_df):
            return [1]

        single_method_call_list = processed_df['method']
        n_gram_method_call_list = []
        num_of_n_gram_method_calls = len(single_method_call_list) - n_gram + 1
        for i in range(num_of_n_gram_method_calls):
            my_n_gram_method_call = ''
            for j in range(i, i+n_gram):
                my_n_gram_method_call += single_method_call_list[j]
            n_gram_method_call_list.append(my_n_gram_method_call)

        unique_n_gram_method_call_list = list(set(n_gram_method_call_list))
        unique_n_gram_method_call_list.sort(key=n_gram_method_call_list.index)
        attribte_n_gram = []
        for unqiue_n_gram_method_call in unique_n_gram_method_call_list:
            call_times = n_gram_method_call_list.count(unqiue_n_gram_method_call)
            attribte_n_gram.append(call_times/len(n_gram_method_call_list))

        return attribte_n_gram

    def get_time_series_of_all_attributes(self):
        sample_vector_list = []
        for current_tree_level in range(1, self.tree_level+1):
            for n_gram in range(1, self.n_gram+1):
                current_level_nodes = get_nodes_at_specific_level(self.method_call_tree, current_tree_level)
                current_processed_df = self._get_processed_df_at_specific_level(current_level_nodes)
                # print_full_dataframe(current_processed_df)
                current_n_gram = self._get_attribute_n_gram_of_method_calls(current_processed_df, n_gram)
                sample_vector_list.append(current_n_gram)

        print('Start validation on sample_vector_list')
        for vector in sample_vector_list:
            if len(vector) == 0:
                raise ValueError('Any vector must not be an empty list!')
        print('Validation completed on sample_vector_list')

        return sample_vector_list

if __name__ == '__main__':
    import utilities.Constants as Constants
    from representors.level_histo_representor import Level_Histo_Representor

    level_histo_param_dict = {'tree_level_list': Constants.TREE_LEVEL_LIST,
                              'fixed_length_list': Constants.FIXED_LENGTH_LIST,
                              'gram_num_list': Constants.GRAM_NUMBER_LIST,
                              'etl_component': Constants.MY_ETL_COMPONENTS[1]}
    my_generator = Level_Histo_Representor({}, 'Test', level_histo_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()
    print()