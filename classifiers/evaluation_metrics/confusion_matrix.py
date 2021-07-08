from sklearn.metrics import confusion_matrix
import numpy as np
class Confusion_Accuracy:
    def __init__(self, y_test_pred, y_test):
        if not isinstance(y_test_pred, list) or not isinstance(y_test, list):
            raise ValueError("y_test_pred and y_test must be of the list type!")
        if len(y_test_pred) != len(y_test):
            raise ValueError("y_test_pred and y_test must be of the same length!")
        self.y_test_pred = y_test_pred
        self.y_test = y_test

    def get_evaluation_metric(self):
        allLabelsList=np.unique(np.concatenate((self.y_test, self.y_test_pred), axis=0))
        print(allLabelsList)
        confusionMatrix = confusion_matrix(self.y_test,self.y_test_pred, labels=allLabelsList)
        return confusionMatrix