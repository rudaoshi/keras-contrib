





class Problem(object):


    def __init__(self, corpus):

        self.corpus = corpus

    def is_supervised(self):

        raise NotImplementedError

    def samples(self):
        raise NotImplementedError

    @staticmethod
    def objective(label, pred):
        raise NotImplementedError