class RTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
    
    def author(self):
        return "aperez374"
    
    def study_group(self):
        return "aperez374"
    
    def add_evidence(self, data_x, data_y):
        pass
    
    def query(self, points):
        pass
    
if __name__ == "__main__":
    print("This is a RTLearner")