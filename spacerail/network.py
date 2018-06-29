from collections import namedtuple
from collections import abc
from spacerail.base import *

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

    def __add__(self,other):
        if isinstance(other, Path):
            return Path(self._items + other._items)
        else:
            return Path(self._items + other)

    @property
    def end(self):
        return self._items[-1]

    @property
    def tags(self):
        return [x.tag for x in self._items if isinstance(x, EdgeData) and x.tag is not None]

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

class PathSet(ExtendSet):
    def _repr_html_(self):
        o = next(elem.objects[0] for path in self for elem in path._items if isinstance(elem,Node) and len(elem.objects) > 0)
        return o._inf.vis.paths(self)


class Network:
    def __init__(self, nodes=None):
        self.nodes = nodes
        if self.nodes is None: self.nodes = []

    def paths(self, start=None, to=None, directed=True):
        return None

    def search(self, start_node, is_goal_node, max_dist, min_dist):
        stack = [(start_node, Path([start_node]))]
        while stack:
            node,path = stack.pop()
            for e in node.edges:
                new_path = path + [e, e.goal, e.goal.other]
                if is_goal_node(e.goal, RelativeDir.OPPOSITE) and min_dist <= new_path.length <= max_dist:
                    yield new_path
                else:
                    if is_goal_node(e.goal.other, RelativeDir.SAME) and min_dist <= new_path.length <= max_dist:
                        yield new_path
                    elif new_path.length <= max_dist:
                        stack.append((e.goal.other, new_path))


    def mk_node_pair(self,t,pos,dir):
        a = Node()
        b = Node()
        a.other = b
        b.other = a
        self.nodes.append(a)
        self.nodes.append(b)
        a.network = self
        b.network = self

        a.pos = (t,pos,-dir)
        b.pos = (t,pos,dir)

        return a,b
