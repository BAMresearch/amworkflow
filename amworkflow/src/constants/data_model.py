import numpy as np

class MapParamModel(object):
    def __init__(self, label: list, data: list = None):
        self.data = data if isinstance(data, list) or isinstance(data, np.ndarray) else [None for i in range(len(label))]
        self.label = label
        self.mapping()
        self.dict = dict(zip(self.label, self.data))
    
    def mapping(self):
        for ind, lbl in enumerate(self.label):
            setattr(self, lbl, self.data[ind])
    
class DeepMapParamModel(object):
    def __init__(self, target_dict: dict):
        self.d = target_dict
        for key, value in self.d.items():
            if isinstance(value, dict):
                setattr(self, key, DeepMapParamModel(value))
            else:
                setattr(self, key, value)

