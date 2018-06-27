from collections import namedtuple
from collections import abc

class EdgeData(namedtuple("EdgeData",["origin","goal","length","tag"])):
    def truncate(self, offset, length):
        return PartialEdge(offset, length, self)

class PartialEdge(namedtuple("PartialEdge",["offset","dist","edgedata"])): 
    @property
    def length(self):
        return self.edgedata.length - self.dist - self.offset

    def truncate(self,length):
        return self.edgedata.truncate(offset, length)

class Node:
    @property
    def length(self):
        return 0.0

    def __init__(self, other=None, edges=None, objects=None):
        self.other = other
        self.edges = edges
        if self.edges is None: self.edges = []
        self.objects = objects
        if self.objects is None: self.objects = []

class Path:
    def __init__(self, items):
        self._items = items

    @property
    def length(self):
        return sum(x.length for x in self._items)

    def truncate(self, new_length):
        new_items = list(self._items)
        length_diff = self.length - new_length
        while length_diff > 0.0 and new_items:
            remove_item = new_items.pop()
            if remove_item.length <= length_diff:
                length_diff -= remove_item.length
            else:
                new_items.append(remove_item.truncate(length_diff))
                length_diff = 0.0
        return Path(new_items)

class PathSet:
    pass

class Network:
    def __init__(self, nodes=None):
        self.nodes = nodes
        if self.nodes is None: self.nodes = []

    def paths(self, start=None, to=None, directed=True):
        return None


    def mk_node_pair(self):
        a = Node()
        b = Node()
        a.other = b
        b.other = a
        self.nodes.append(a)
        self.nodes.append(b)
        return a,b
