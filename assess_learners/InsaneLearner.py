import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=verbose) for _ in range(20)]
    def author(self):
        return "aperez374"
    def study_group(self):
        return "aperez374"
    def add_evidence(self, dataX, dataY):
        for learner in self.learners:
            learner.add_evidence(dataX, dataY)
    def query(self, points):
        return np.mean(np.array([learner.query(points) for learner in self.learners]), axis=0)

if __name__ == "__main__":
    print("This is an Insane Learner")