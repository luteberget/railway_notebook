from enum import Enum
from collections import abc

class ExtendSet(abc.Set, abc.Hashable):
    def __init__(self, l): self._items = frozenset(l)
    def __contains__(self,item): return item in self._items
    def __iter__(self): return self._items.__iter__()
    def __len__(self): return len(self._items)
    def __hash__(self): return self._items.__hash__()

class RelativeDir(Enum):
    SAME = 1
    OPPOSITE = 2

class Dir(Enum):
    """Direction up/down."""
    UP = 1
    DOWN = 2

    def __neg__(self): 
        """Return the opposite direction"""
        return self.opposite()

    @staticmethod
    def from_string(s):
        """Parse a string into Dir ("up" -> Dir.UP, "down" -> Dir.DOWN)"""
        if s.lower() == "up": return Dir.UP
        if s.lower() == "down": return Dir.DOWN
        raise Exception("Unknown direction value: \"{}\"".format(s))

    def opposite(self):
        """Return the opposite direction"""
        if self == Dir.UP: return Dir.DOWN
        if self == Dir.DOWN: return Dir.UP
        raise Exception("Unknown direction value: \"{}\"".format(self))

    def factor(self):
        """Returns the number 1 for Dir.UP and -1 for Dir.DOWN"""
        if self == Dir.UP: return 1
        if self == Dir.DOWN: return -1
        raise Exception("Unknown direction value: \"{}\"".format(self))

class Side(Enum):
    LEFT = 1
    RIGHT = 2

    def opposite(self):
        if self == Side.LEFT: return Side.RIGHT
        if self == Side.RIGHT: return Side.LEFT
        raise Exception("Unknown side value: \"{}\"".format(self))
