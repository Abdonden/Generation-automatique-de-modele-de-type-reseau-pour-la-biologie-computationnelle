
from torch_geometric.data import Data
import torch

class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key=="y":
            return self.num_nodes
        return super().__inc__(key,value,*args,**kwargs)
 
