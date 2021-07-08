class Evaluation_Accuracy:
    def __init__(self, y_test_pred, y_test):
        if not isinstance(y_test_pred, list) or not isinstance(y_test, list):
            raise ValueError("y_test_pred and y_test must be of the list type!")
        if len(y_test_pred) != len(y_test):
            raise ValueError("y_test_pred and y_test must be of the same length!")
        self.y_test_pred = y_test_pred
        self.y_test = y_test

    def get_evaluation_metric(self):
        hit_num = 0
        for i in range(len(self.y_test_pred)):
            if self.y_test_pred[i] == self.y_test[i]:
                hit_num += 1
        return hit_num/len(self.y_test)