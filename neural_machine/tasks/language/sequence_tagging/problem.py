
from neural_machine.tasks.language.common.problem import Problem

class SequenceTaggingTrainingProblem(Problem):

    def __init__(self, corpus):
        super(SequenceTaggingTrainingProblem, self).__init__(corpus)

    def is_supervised(self):
        return True

    def data_names(self):
        return ["data"]

    def label_names(self):

        return ["label"]

    def samples(self):

        for sample, label in self.corpus.corpus:
            yield [sample], [label]


