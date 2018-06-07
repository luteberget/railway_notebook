# -*- coding: utf-8 -*-
"""rc -- railway analysis tools

This module exposes the basic building blocks for automating creation of interlocking
tables for railway control systems.
"""

import xml.etree.ElementTree
import numbers
import math
from math import inf
from enum import Enum
from collections import namedtuple
from collections import abc

import pandas as pd
tbl = pd.DataFrame

empty = set()

class _Set(abc.Set, abc.Hashable):
    def __init__(self, l): self._items = frozenset(l)
    def __contains__(self,item): return item in self._items
    def __iter__(self): return self._items.__iter__()
    def __len__(self): return len(self._items)
    def __hash__(self): return self._items.__hash__()

class AreaSet(_Set):
    def from_delimiters(set):
        return AreaSet(_mk_sections(set))

    def filter(self, func=None):
        if func is None: func = lambda x: True
        return AreaSet(filter(func, self._items))

    def __str__(self): return "Set of {} areas.".format(len(self._items))

def _mk_sections(objects):
    delims = set([Delimiter(o,dir) for o in objects for dir in [Dir.UP, Dir.DOWN]])
    while delims:
        delim = delims.pop()
        section = Area.empty()
        finished_delims = set([delim])
        new_delims = set([delim])
        while new_delims:
            delim= new_delims.pop()
            for path in delim.object.find(delim.dir,objects):
                if not (path.end, path.dir.opposite()) in finished_delims:
                    new = Delimiter(path.end, path.dir.opposite())
                    finished_delims.add(new)
                    new_delims.add(new)
                    delims.remove(new)
                    section = section + path.to_area()
        yield section

class IntervalSet(_Set):
    def __init__(self, xs):
        xs = list(xs)
        items = []
        for t in frozenset([i.track for i in xs]):
            ys = list(sorted([[i.up().pos_a, i.up().pos_b] for i in xs if i.track == t], key = lambda x: x[0]))
            t_is = [ys[0]]
            for x in ys[1:]:
                if t_is[-1][1] < x[0]:
                    t_is.append(x)
                else:
                    t_is[-1][1] = max(t_is[-1][1], x[1])
            items += list(map(lambda x: TrackInterval(t, x[0], x[1]), t_is))
        super().__init__(items)

    def __le__(self, other): return self.__eq__(other) or self.__lt__(other)
    def __ge__(self, other): return self.__eq__(other) or self.__gt__(other)
    def __gt__(self,other): return other.__lt__(self)

    def __eq__(self,other): 
        return list(sorted(self._items)) == list(sorted(other._items))

    def __ne__(self,other): return not (self == other)

    def __hash__(self): return self._items.__hash__()

    def __lt__(self, other):
        strict = False
        for t in frozenset([x.track for x in self._items]):
            ais = [(x.pos_a, x.pos_b) for x in self._items if x.track == t]
            bis = [(x.pos_a, x.pos_b) for x in other._items if x.track == t]

            all_ps = [p for p in [x.pos_a, x.pos_b] for x in ais] + \
                     [p for p in [x.pos_a, x.pos_b] for x in bis]

            ais = frozenset([i for i in interval_split(a,b,all_ps) for a,b in ais])
            bis = frozenset([i for i in interval_split(a,b,all_ps) for a,b in bis])

            if ais > bis: return False
            if ais < bis: strict = True

        return strict

def interval_split(a,b,split_ps):
    ps = [a] + [s for s in sorted(split_ps) if a < s < b] + [b]
    return [(p1,p2) for p1,p2 in zip(ps,ps[1:])]


class DelimiterSet(_Set):
    pass


class SwitchState(namedtuple('SwitchState',["object","side"])): 
    def reversed(self): return self

class Delimiter(namedtuple('Delimiter',["object","dir"])): pass
class UnknownConnection(namedtuple('UnknownConnection',[])): pass

class Area(namedtuple('Area',["delimiters","intervals"])):
    def empty(): return Area(DelimiterSet([]),IntervalSet([]))

    def __add__(self, other):
        return Area(self.delimiters | other.delimiters, self.intervals | other.intervals)

    def __and__(self,other):
        return self.intervals & other.intervals

class Path(abc.Collection):
    def __init__(self, seq):
        self._items = list(seq)

    def interval(interval): return Path([interval])
    def unknownconnection(): return Path([UnknownConnection()])
    def switchstate(state): return Path([state])
    def empty(): return Path([])

    def __add__(self, other):
        if not self._items: return other
        if not other._items: return self

        prev = self._items[-1]
        next = other._items[0]

        if isinstance(prev,DirectedTrackInterval) and isinstance(next,DirectedTrackInterval):
            return Path(self._items[:-1] + [prev.append(next)] + other._items[1:])
        else:
            return Path(self._items + other._items)

    def reversed(self):
        return Path(map(lambda x: x.reversed(), reversed(self._items)))

    def length(self):
        return sum([e.length() for e in self._items if isinstance(e, DirectedTrackInterval)])

    def to_intervalset(self):
        return IntervalSet([e for e in self._items if isinstance(e, DirectedTrackInterval)])

    def __contains__(self, item): return self._items.__contains__(item)
    def __iter__(self): return self._items.__iter__()
    def __len__(self): return len(self._items)

    def __str__(self):
        return "Path with {} elements, length {} m".format(len(self._items), self.length())
        

class DelimitedPath(namedtuple('DelimitedPath',["dir", "start","links","end"])):
    def to_area(self):
        delimiters = DelimiterSet([Delimiter(self.start,self.dir),
                                   Delimiter(self.end,  self.dir.opposite())])
        return Area(delimiters, self.links.to_intervalset())

class TrackRef(namedtuple('TrackRef',["model","id","name"])):
    def __str__(self):
        return "Track: id={} name=\"{}\"".format(self.id, self.name)
    def short_repr(self):
        if len(self.name) == 0:
            return "TrackId({})".format(self.id)
        elif len(self.name) < 25:
            return "{}".format(self.name)
        else:
            return "{}...".format(self.name[:25])
    def length(self):
        return _track_length(self.model._xml_tracks[self.id])
    def _from_xml(model,e):
        return TrackRef(model, e.attrib["id"],e.attrib["name"])
    def at_pos(self,pos):
        return TrackPos(self,pos)

class TrackInterval(namedtuple('TrackInterval',["track","pos_a","pos_b"])):
    def length(self):
        return abs(self.pos_a - self.pos_b)

    def up(self): return self

class DirectedTrackInterval(namedtuple('DirectedTrackInterval',["track","pos_a","pos_b"])):
    def reversed(self):
        return DirectedTrackInterval(self.track, self.pos_b, self.pos_a)

    def length(self):
        return abs(self.pos_a - self.pos_b)

    def append(self, other):
        if not (self.pos_b == other.pos_a and self.track == other.track):
            raise Exception("Cannot append intervals {} {}".format(self,other))
        return DirectedTrackInterval(self.track,self.pos_a,other.pos_b)

    def contains(self,other):
        return (self.pos.track == other.track and
                self.pos.pos <= other.pos and
                self.pos.pos + self.length >= other.pos)

    def up(self):
        if self.pos_a > self.pos_b:
            return TrackInterval(self.track, self.pos_b, self.pos_a)
        else:
            return TrackInterval(self.track, self.pos_a, self.pos_b)

class TrackPos(namedtuple('TrackPos',["track","pos"])):
    """Position on a track in the input model"""
    def __str__(self):
        return "Position: track=\"{}\" pos={}".format(self.track.short_repr(),self.pos)

    def __add__(self, x):
        if isinstance(x, numbers.Number):
            return self.track.model.translate_pos(self, x)
        else:
            raise ArithmeticError("RHS operand must be a number")

    def __sub__(self, x):
        if isinstance(x, numbers.Number):
            return self.track.model.translate_pos(self, -x)
        else:
            raise ArithmeticError("RHS operand must be a number")

    def to_pos(self,pos_b):
        return DirectedTrackInterval(self.track, self.pos, pos_b)

TrackPos.track.__doc__ = "Track reference"
TrackPos.pos.__doc__ = "Distance from start of track"

class Dir(Enum):
    UP = 1
    DOWN = 2

    def from_string(s):
        if s.lower() == "up": return Dir.UP
        if s.lower() == "down": return Dir.DOWN
        raise Exception("Unknown direction value: \"{}\"".format(s))

    def opposite(self):
        if self == Dir.UP: return Dir.DOWN
        if self == Dir.DOWN: return Dir.UP
        raise Exception("Unknown direction value: \"{}\"".format(self))

    def factor(self):
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

class DirectedTrackPos(namedtuple('DirectedTrackPos',["trackpos","dir"])):
    pass


ns = {'railml': 'http://www.railml.org/schemas/2013',
        'dc': 'http://purl.org/dc/elements/1.1/'
        }

def _partition(l,pred):
    a,b = [],[]
    for x in l:
        if pred(x): a.append(x)
        else: b.append(x)
    return (a,b)

def _sorted_pos_xml(l):
    return sorted(l, key=lambda x: float(x.attrib["pos"]))

def _sorted_pos(l):
    return sorted(l, key=lambda x: x.pos())

objectelementnames = [ 
        ("trackTopology", [
            ("connections","switch"),
            ("connections","crossing"),
            ("mileageChanges","mileageChange"),
            ]),
        ("trackElements", [
            ("radiusChanges","radiusChange"),
            ("gradientChanges","gradientChange"),
            ]),
        ("ocsElements", [
            ("signals","signal"),
            ("trainDetectionElements","trainDetector"),
            ("derailers","derailer"),
            ]),
        ]

def _track_objects(e):
    objs = []
    topo = e.find("railml:trackTopology",ns)
    objs.append(topo.find("railml:trackBegin",ns))
    objs.append(topo.find("railml:trackEnd",ns))
    for (toplevelcontainername, subcontainers) in objectelementnames:
        toplevelelement = e.find("railml:{}".format(toplevelcontainername), ns)
        if not toplevelelement: continue
        for (containername,elementname) in subcontainers:
            container = toplevelelement.find("railml:{}".format(containername), ns)
            if not container: continue
            objs += list(container.findall("railml:{}".format(elementname), ns))
    return objs

def _track_length(e):
    topo = e.find("railml:trackTopology",ns)

    begin = topo.find("railml:trackBegin",ns)
    end   = topo.find("railml:trackEnd",ns)

    begin_pos = float(begin.attrib["pos"])
    end_pos   = float(  end.attrib["pos"])

    return end_pos - begin_pos

class Model:
    """Railway infrastructure model (tracks and trackside objects)"""

    def find_paths_directed(self, dir, start_obj, goal_set, max_dist, min_dist):
        stack = [(start_obj, Path.empty())]
        while stack:
            obj,links = stack.pop()
            for next_obj, tags in obj.get_links(dir):
                new_path = links + tags
                if next_obj in goal_set and min_dist <= new_path.length() <= max_dist:
                    yield DelimitedPath(dir, start_obj, new_path, next_obj)
                elif new_path.length() <= max_dist:
                    stack.append((next_obj, new_path))


    def __init__(self, filename):
        """Read infrastructure model from XML file."""

        # Load infrastructure from file
        tree = xml.etree.ElementTree.parse(filename).getroot()
        iss = list(tree.findall('railml:infrastructure',ns))
        if len(iss) != 1: raise Exception("Infrastructure not found in file.")
        self._infrastructure = iss[0]

        # Tracks
        tracks = list(self._infrastructure.findall('railml:tracks',ns))
        if len(tracks) != 1: raise Exception("Infrastructure contains no tracks.")
        tracks = tracks[0]
        tracks = list(tracks.findall('railml:track',ns))
        if len(tracks) == 0: raise Exception("Infrastructure contains no tracks.")

        self._xml_tracks = { (t.attrib["id"]):t for t in  tracks }
        self.tracks = [TrackRef(self, e.attrib["id"],e.attrib["name"]) for e in tracks]

        # Topology
        self._build_graph()

    def _build_graph(self):
        self.objects = []
        conn_id_objects = {}
        for t in self._xml_tracks.values():
            objs = [PointObject(self,t,x) for x in _sorted_pos_xml(_track_objects(t)) ]
            self.objects += objs
            for o in objs:
                for (a,b,d) in _object_connections(o._xml):
                    conn_id_objects[a] = (b,d,o)

            for o1, o2 in zip(objs, objs[1:]):
                path = Path.interval(o1.pos().to_pos(o2.pos().pos))
                if "switch" in o1._xml.tag and _switch_orientation(o1._xml) == Dir.UP: 
                    path = Path.switchstate(SwitchState(o1, _switch_connection_course(o1._xml).opposite())) + path
                if "switch" in o2._xml.tag and _switch_orientation(o2._xml) == Dir.DOWN:
                    path = path + Path.switchstate(SwitchState(o2, _switch_connection_course(o2._xml).opposite()))
                o1.up_links.append((o2,path))
                o2.down_links.append((o1,path.reversed()))

        # resolve connections
        while conn_id_objects:
            id = next(iter(conn_id_objects))
            ref,dir,o1 = conn_id_objects[id]
            del conn_id_objects[id]
            id2,dir2,o2 = conn_id_objects[ref]
            del conn_id_objects[ref]

            assert id == id2
            assert dir.opposite() == dir2

            path = Path.empty()
            if "switch" in o1._xml.tag and _switch_orientation(o1._xml) == dir:
                path = Path.switchstate(SwitchState(o1, _switch_connection_course(o1._xml))) + path
            elif "switch" in o2._xml.tag and _switch_orientation(o2._xml) == dir.opposite():
                path = path + Path.switchstate(SwitchState(o2, _switch_connection_course(o2._xml)))
            else:
                path = path + Path.unknownconnection()

            if dir == Dir.UP:
                o1.up_links.append((o2,path))
                o2.down_links.append((o1,path.reversed()))
            if dir == Dir.DOWN:
                o1.down_links.append((o2,path))
                o2.up_links.append((o1,path.reversed()))

        self.objects = PointObjectSet(self.objects)

    def translate_pos(self, pos,l):
        return [pos.pos + l,0.0]

def _switch_orientation(e):
    conn = e.find("railml:connection",ns)
    if conn.attrib["orientation"] == "outgoing": return Dir.UP
    if conn.attrib["orientation"] == "incoming": return Dir.DOWN
    raise Exception("Unknown switch orientation {}".format(e))

def _switch_connection_course(e):
    conn = e.find("railml:connection",ns)
    if conn.attrib["course"] == "left": return Side.LEFT
    if conn.attrib["course"] == "right": return Side.RIGHT
    raise Exception("Unknown switch course {}".format(e))

def _object_connections(e):
    for conn in e.findall("railml:connection",ns):
        dir = None
        if "trackBegin" in e.tag: dir = Dir.DOWN
        if "trackEnd" in e.tag: dir = Dir.UP
        if "switch" in e.tag: dir = _switch_orientation(e)
        yield (conn.attrib["id"], conn.attrib["ref"],dir)

class PointObject:
    def find_forward(self, set=None, max_dist=inf, min_dist=0.0):
        return self.model.find_paths_directed(self.dir(), self, set, max_dist, min_dist)

    def find(self, dir, set=None, max_dist=inf, min_dist=0.0):
        return self.model.find_paths_directed(dir, self, set, max_dist, min_dist)

    def pos(self):
        return TrackRef._from_xml(self.model,self._xml_track).at_pos(float(self._xml.attrib["pos"]))

    def dir(self):
        return Dir.from_string(self._xml.attrib["dir"])

    def __getattr__(self, name):
        try:
            return self._xml.attrib[name]
        except KeyError:
            return None

    def __str__(self):
        if self.name: return "Point object name='{}' id={}".format(self.name, self.id)
        if self.code: return "Point object code='{}' id={}".format(self.code, self.id)
        return "Point object id='{}'".format(self.id)

    def _repr_html_(self): return self.__str__()

    def __init__(self, model, xml_track, xml):
        self.model = model
        self._xml = xml
        self._xml_track = xml_track
        self.up_links = []
        self.down_links = []

    def get_links(self, dir):
        if dir == Dir.UP: return self.up_links
        if dir == Dir.DOWN: return self.down_links
        raise Exception()


#TODO check type of objects in constructor?
class PointObjectSet(_Set):
    def __str__(self): return "Set of {} point objects.".format(len(self._items))

    def find(self,func=lambda x: True, type=None, dir=None, location=None, set=None,id=None):
        return next(iter(self.filter(func, type, dir, location, set,id)), None)

    def filter(self,func=lambda x: True, type=None, dir=None, location=None, set=None,id=None):
        base = self
        if type is not None:
            base = filter(lambda x: type.lower() in x._xml.tag.lower(), base)
        if id is not None:
            base = filter(lambda x: id == x.id, base)
        return PointObjectSet(filter(func, base))

    def find_forward(self, set=None, max_dist=inf, min_dist=0.0):
        if not callable(set): set = lambda x: set
        return [path for start in self \
                     for path in start.find_forward(set(start), max_dist, min_dist)]

    def _repr_html_(self):
        return tbl(list(self._items))._repr_html_()

