from annoy import AnnoyIndex


class AnnoyVectorDB:
    def __init__(self):
        pass

    def build(self, f: int, metric: str, n_trees: int, items: list):
        """
        Build the AnnoyIndex with the given parameters
        :param f: Length of item vector that will be indexed
        :param n_trees: Number of trees to build
        :param items: List of vectors to be indexed
        """
        self.t = AnnoyIndex(f, metric)
        for i, item in enumerate(items):
            self.t.add_item(i, item)
        self.t.build(n_trees)
        self.t.save("test.ann")
