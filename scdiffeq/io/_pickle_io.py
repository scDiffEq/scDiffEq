
import pickle


class PickleIO:
    def __init__(self):
        pass

    def read(self, path, mode="rb"):
        return pickle.load(self.__path__(path, mode))

    def write(self, obj, path, mode="wb", protocol=pickle.HIGHEST_PROTOCOL):
        """If writing for use in colab, use protocol=4"""

        pickle.dump(obj=obj, file=self.__path__(path, mode), protocol=protocol)

    def __path__(self, path, mode):
        return open(path, mode)
    
    
def read_pickle(path, mode="rb"):

    pickle_io = PickleIO()
    return pickle_io.read(path, mode=mode)


def write_pickle(obj, path, mode="wb", protocol=pickle.HIGHEST_PROTOCOL):

    pickle_io = PickleIO()
    return pickle_io.write(obj=obj, path=path, mode=mode, protocol=protocol)
