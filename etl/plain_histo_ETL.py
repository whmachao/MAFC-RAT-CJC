# TEST
import numpy as np
import pandas as pd
from treelib import Node

from utilities.Utils import new_process_single_trace


class Plain_Histo_Extractor:
    def __init__(self, method_call_tree, root_at_top_level):
        if len(root_at_top_level) != 1:
            raise ValueError('nodes_at_specific_level MUST have only one element!')

        self.node = root_at_top_level[0]
        if not isinstance(self.node, Node):
            raise ValueError('self.node is NOT of Tree type!')

        if not isinstance(self.node.data[0], pd.DataFrame) or not isinstance(self.node.data[1], pd.DataFrame) or not isinstance(self.node.data[2], pd.DataFrame):
            raise ValueError('self.node.data contains some non-Dataframe element!')

        if self.node.data[0].empty or self.node.data[1].empty or self.node.data[2].empty:
            raise ValueError('self.node.data contains some EMPTY Dataframe element!')

        self.thread_one_df = self.node.data[0]
        self.event_eight_df = self.node.data[1]
        self.event_nine_df = self.node.data[2]
        self.processed_df = new_process_single_trace(self.thread_one_df, self.event_eight_df)

    def _get_attribute_thread_clock_duration_of_each_method_call(self):
        attribteOne_vector = []
        for index, row in self.processed_df.iterrows():
            start_clock, end_clock = int(row['start_clock']), int(row['end_clock'])
            attribteOne_vector.append(end_clock - start_clock)

        return attribteOne_vector

    def _get_attribute_count_methods(self):
        attributeTwo_vector = []
        distinctMethodsList=[]
        count=0
        for index, row in self.processed_df.iterrows():
            # count+=1
            if row['method'] not in distinctMethodsList:
                distinctMethodsList.append(row['method'])
                attributeTwo_vector.append(1)
            else:
                theIndex=distinctMethodsList.index(row['method'])
                attributeTwo_vector[theIndex]+=1
        # print(count)
        return attributeTwo_vector


    def _get_attribute_get_proportion_of_two_adjacent_methods(self):
        methodList = []
        methodsGroupList = []
        methodsGroupCount = []

        for index, row in self.processed_df.iterrows():
            if 'method' in row.keys():
                methodList.append(row['method'])
        for i in range(len(methodList) - 2 + 1):
            tmp = []
            for j in range(2):
                tmp.append(methodList[i + j])
            # methodsGroupList.append(tmp)
            if tmp not in methodsGroupList:
                methodsGroupList.append(tmp)
                methodsGroupCount.append(1)
            else:
                theIndex = methodsGroupList.index(tmp)
                methodsGroupCount[theIndex] = methodsGroupCount[theIndex] + 1

        methodsCountList=methodsGroupCount
        totalTimes = sum(methodsCountList)
        attribteThree_vector = []
        for i in range(len(methodsCountList)):
            attribteThree_vector.append(methodsCountList[i] / totalTimes)
        return attribteThree_vector

    def _get_attribute_time_variance_of_adjacent_methods(self):
        threadClockList = []
        methodList = []
        # print(preprocessed_df['start_clock'][0])
        for index, row in self.processed_df.iterrows():
            if 'method' in row.keys():
                methodList.append(row['method'])
        processedMethod = []
        varianceList = []
        # print(len(methodList))
        for i in range(len(methodList) - 1):
            headMethod = methodList[i]
            tailMethod = methodList[i + 1]
            timeList = []
            if [headMethod, tailMethod] not in processedMethod:
                for j in range(len(methodList) - 1):
                    if methodList[j] == headMethod and methodList[j + 1] == tailMethod:
                        timeList.append(self.processed_df['end_clock'][j] - self.processed_df['start_clock'][j])
                theVariance = np.var(timeList)
                varianceList.append(theVariance)
                processedMethod.append([headMethod, tailMethod])
        attributeFour_vector=varianceList
        return attributeFour_vector


    def _get_attribute_probablity_list(self):
        methodList=[]
        for index, row in self.processed_df.iterrows():
            methodList.append(row['method'])

        headMethodTotalNum=[] # 记录以某个method为头部的序列对出现的次数
        distinctMethodList=[]
        for i in range(len(methodList)-1):
            if methodList[i] not in distinctMethodList:
                distinctMethodList.append(methodList[i])
                headMethodTotalNum.append(1)
            else:
                theIndex=distinctMethodList.index(methodList[i])
                headMethodTotalNum[theIndex]+=1

        attributeFive_vector=[]
        for i in range(len(methodList)-1):
            headMethod=methodList[i]
            tailMethod=methodList[i+1]
            count=0
            for j in range(len(methodList)-1):
                if methodList[j]==headMethod and methodList[j+1]==tailMethod:
                    count+=1
            theIndex=distinctMethodList.index(headMethod)
            totalNum=headMethodTotalNum[theIndex]
            attributeFive_vector.append(float(count)/totalNum)
        return attributeFive_vector


    # def _get_attribute_method2vec(self):
    #     thread_one_df, event_eight_df, event_nine_df = get_dataframe_from_jsontrace(self.trace_Path)
    #
    #     methodName = []
    #     retVal = []
    #     for index, row in event_eight_df.iterrows():
    #         methodName.append(row['method'])
    #         retVal.append(row['retVal'])
    #
    #     threadMethodList = []
    #     for index, row in thread_one_df.iterrows():
    #         threadMethodList.append(row['method'])
    #
    #     threadNameList = []
    #     for item in threadMethodList:
    #         if item in retVal:
    #             theIndex = retVal.index(item)
    #             threadNameList.append(methodName[theIndex])
    #         else:
    #             raise ValueError('Error! can not find the name of method')
    #
    #     for i in range(len(threadNameList)):
    #         threadNameList[i]=threadNameList[i].replace(" ","")
    #         if len(threadNameList[i])>98:
    #             threadNameList[i]=threadNameList[i][0:98]
    #
    #     vector_List, methodVec = [], None
    #     model = word2vec.load("vector.bin",vocabUnicodeSize=500)
    #     for item in threadNameList:
    #         methodVec=model[item].tolist().copy()
    #         vector_List.append(methodVec)
    #
    #     return methodVec

    def get_time_series_of_all_attributes(self):
        ts_of_attribute_one = self._get_attribute_thread_clock_duration_of_each_method_call()
        ts_of_attribute_two = self._get_attribute_time_variance_of_adjacent_methods()
        ts_of_attribute_three = self._get_attribute_count_methods()
        ts_of_attribute_four = self._get_attribute_get_proportion_of_two_adjacent_methods()
        ts_of_attribute_five = self._get_attribute_time_variance_of_adjacent_methods()
        ts_of_attribute_six = self._get_attribute_probablity_list()
        # ts_of_attribute_seven = self._get_attribute_method2vec()
        sample_vector_list = []
        sample_vector_list.append(ts_of_attribute_one)
        sample_vector_list.append(ts_of_attribute_two)
        sample_vector_list.append(ts_of_attribute_three)
        sample_vector_list.append(ts_of_attribute_four)
        sample_vector_list.append(ts_of_attribute_five)
        sample_vector_list.append(ts_of_attribute_six)
        # sample_vector_list.append(ts_of_attribute_seven)

        print('Start validation on sample_vector_list')
        for vector in sample_vector_list:
            if len(vector) == 0:
                raise ValueError('Any vector must not be an empty list!')
        print('Validation completed on sample_vector_list')

        return sample_vector_list

if __name__ == '__main__':
    import utilities.Constants as Constants
    from representors.plain_histo_representor import Plain_Histo_Representor

    plain_histo_param_dict = {'tree_level_list': None,
                              'bin_num_list': Constants.BIN_NUMBER_LIST,
                              'etl_component': Constants.MY_ETL_COMPONENTS[0]}
    my_generator = Plain_Histo_Representor({}, 'Test', plain_histo_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()
    print()