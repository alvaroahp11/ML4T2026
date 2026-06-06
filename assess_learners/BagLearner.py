class BagLearner:
    def __init__(self, learner, kwargs, bags, verbose=False):
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))
        self.verbose = verbose

    def author(self):
        return "aperez374"
    
    def study_group(self):
        return "aperez374"
        

    def addEvidence(self, dataX, dataY):
        for learner in self.learners:
            learner.addEvidence(dataX, dataY)

    def query(self, points):
        predictions = []
        for learner in self.learners:
            predictions.append(learner.query(points))
        return sum(predictions) / len(predictions)

if __name__ == "__main__":
    print("This is a Bag Learner")