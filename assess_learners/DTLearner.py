import numpy as np

class DTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])

    def author(self):
        return "aperez374"
    
    def study_group(self):
        return "aperez374"

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        if len(data_x) <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        
        elif np.all(data_y == data_y[0]):
            return np.array([[-1, data_y[0], np.nan, np.nan]])
        

        correlations = np.zeros(data_x.shape[1])
        for i in range(data_x.shape[1]):
            if np.std(data_x[:, i]) > 0:           
                correlations[i] = np.corrcoef(data_x[:, i], data_y)[0, 1]
           
        correlations = np.nan_to_num(correlations)

        best_feature = np.argmax(np.abs(correlations))
        best_value = np.median(data_x[:, best_feature])

        left_indices = data_x[:, best_feature] <= best_value
        right_indices = data_x[:, best_feature] > best_value

        if left_indices.sum() == 0 or right_indices.sum() == 0:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        left_tree = self.build_tree(data_x[left_indices], data_y[left_indices])
        right_tree = self.build_tree(data_x[right_indices], data_y[right_indices]) 

        root = np.array([[best_feature, best_value, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))


    def query(self, points):
        results = []
        for point in points:
            node_index = 0
            while self.tree[node_index, 0] != -1:
                feature = int(self.tree[node_index, 0])
                value = self.tree[node_index, 1]
                if point[feature] <= value:
                    node_index += int(self.tree[node_index, 2])
                else:
                    node_index += int(self.tree[node_index, 3])
            results.append(self.tree[node_index, 1])
        return np.array(results)

if __name__ == "__main__":
    print("This is a Decision Tree Learner")

    
