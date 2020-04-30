class Cluster:
    def __init__(self):
        self.seq = []
        self.words = {}
        self.sentences = []
        self.ls = 1
        self.representative = None


class NewCluster(Cluster):
    def __init__(self, sentence):
        super().__init__()
        self.seq = [str(sentence.reviewNumber) + str(sentence.number)]
        self.words = sentence.words
        self.sentences = [sentence]
        self.representative = sentence


class MergedCluster(Cluster):
    def __init__(self, cluster1, cluster2):
        super().__init__()
        for i in cluster2.words:
            if i not in cluster1.words:
                self.words[i] = cluster2.words[i]
            else:
                self.words[i] = cluster1.words[i] + cluster2.words[i]
        for i in cluster1.words:
            if i not in cluster2.words:
                self.words[i] = cluster1.words[i]
        self.seq = cluster1.seq + cluster2.seq
        self.sentences = cluster1.sentences + cluster2.sentences
        if len(cluster1.representative.sentence.word_list) > len(cluster2.representative.sentence.word_list):
            self.representative = cluster1.representative
        else:
            self.representative = cluster2.representative
