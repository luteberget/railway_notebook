# -*- coding: utf-8 -*-
"""rc -- railway analysis tools

This module exposes the basic building blocks for automating creation of interlocking
tables for railway control systems.
"""

import xml.etree.ElementTree
import numbers
import math
from enum import Enum
from collections import namedtuple
from collections import abc

import pandas as pd
tbl = pd.DataFrame

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
        pos_a = self.pos
        if pos_a > pos_b: pos_a,pos_b = pos_b,pos_a
        return TrackInterval(TrackPos(self.track,pos_a), pos_b - pos_a)

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

class TrackInterval(namedtuple('TrackInterval',["pos","length"])):
    def contains(self,other):
        return (self.pos.track == other.track and
                self.pos.pos <= other.pos and
                self.pos.pos + self.length >= other.pos)

ns = {'railml': 'http://www.railml.org/schemas/2013',
        'dc': 'http://purl.org/dc/elements/1.1/'
        }

class Topology(namedtuple('Topology',["segments","links"])):
    def find_paths_directed(self,dir, start_obj, set, max_dist, min_dist):
        stack = [(start_obj,[],0.0)]
        while len(path_stack) > 0:
            obj,links,l = stack.pop()
            for (next_obj,new_links,dl) in obj.next(dir):
                if next_obj in set and min_dist <= l+dl <= max_dist:
                    yield Path(start_obj, links + new_links, next_obj)
                elif l+dl <= max_dist:
                    path_stack.append((next_obj,links+new_links,l+dl))

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

def _track_start_conn_id(t):
    topo = t.find("railml:trackTopology",ns)
    begin = topo.find("railml:trackBegin",ns)
    return begin.attrib["id"]


def _mk_topology2(model,tracks):
    segments = []
    links = []
    link_to_track = []
    start_refs = {}
    for t in tracks:
        start_refs[_track_start_conn_id(t)] = (len(segments)-1,None)
        objs = _track_objects(t)
        pos = 0.0
        l = _track_length(t)
        last_segment = None
        for sw in _sorted_pos_xml(_track_switches(t)): 
            new_pos = float(sw.attrib["pos"])
            (segment_objs,objs) = _partition(objs, 
                    lambda o: pos <= float(o.attrib["pos"]) <= new_pos)
            segments.append((TrackRef._from_xml(model,t).at_pos(pos).to_pos(new_pos), 
                _sorted_pos([PointObject(model,t,obj,len(segments)-1) for obj in segment_objs])))
            if last_segment is not None:
                links.append((last_segment,len(segments)-1))

            last_segment = len(segments)-1
            pos = new_pos
            (conn_dir,conn_id,conn_ref) = _sw_conn(sw)
            if conn_dir == Dir.UP: link_to_track.append((last_segment, conn_ref, sw))
            elif conn_dir == Dir.DOWN: start_refs[conn_id] = (last_segment,sw)

        segments.append((TrackRef._from_xml(model,t).at_pos(pos).to_pos(l), 
                _sorted_pos([PointObject(model,t,obj,len(segments)-1) for obj in objs])))
        if last_segment is not None:
            links.append((last_segment,len(segments)-1))

    return Topology(segments,links)

def _sw_conn(sw):
    conn = sw.find("railml:connection",ns)
    if conn is None: raise Exception("Switch connection missing {}".format(sw)) 

    orientation = conn.attrib["orientation"]
    conn_dir = None
    if orientation == "outgoing": conn_dir = Dir.UP
    if orientation == "incoming": conn_dir = Dir.DOWN
    if conn_dir is None: raise Exception("Switch orientation is missing {}".format(sw))

    return (conn_dir, conn.attrib["id"], conn.attrib["ref"])

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

#def _track_switches(e):
#    topo = e.find("railml:trackTopology",ns)
#    connections = topo.find("railml:connections",ns)
#    sws = []
#    if connections: 
#        for c in connections:
#            if c.tag != "{{{}}}{}".format(ns["railml"],"switch"):
#                raise Exception("Connection type not supported: {}".format(c))
#            sws.append(c)
#    return sws


def _track_length(e):
    topo = e.find("railml:trackTopology",ns)

    begin = topo.find("railml:trackBegin",ns)
    end   = topo.find("railml:trackEnd",ns)

    begin_pos = float(begin.attrib["pos"])
    end_pos   = float(  end.attrib["pos"])

    return end_pos - begin_pos

class Model:
    """Railway infrastructure model (tracks and trackside objects)"""

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
        self.objects = PointObjectSet([o for (_,os) in self.segments for o in os])

    def _build_graph(self):
        segments = []
        conn_id_objects = {}
        for t in self._xml_tracks.values():
            objs = [PointObject(self,x,t) for x in _sorted_pos_xml(_track_objects(t)) ]
            for o in objs:
                for (a,b,d) in _object_connections(o._xml):
                    conn_id_objects[a] = (b,d,o)

            for o1, o2 in zip(objs, objs[1:]):
                edge_tag = None
                dist = (o2.pos() - o1.pos()).length()
                if "switch" in o1._xml.tag and o1._xml.attrib["orientation"] == "outgoing":
                    edge_tag = (o1.id, _switch_continue_course(o1._xml))
                if "switch" in o2._xml.tag and o1._xml.attrib["orientation"] == "incoming":
                    edge_tag = (o2.id, _switch_continue_course(o2._xml))
                o1.up_links.append((o2,dist,edge_tag))
                o2.down_links.append((o1,dist,edge_tag))

        # resolve connections
        while len(conn_id_objects) > 0:
            id = next(iter(conn_id_objects))
            ref,dir,o1 = conn_id_objects[id]
            del conn_id_object[id]
            id2,dir2,o2 = conn_id_objects[ref]
            del conn_id_object[ref]

            assert id == i2
            assert dir.opposite() == dir2

            o1.


        self.segments = segments


    def translate_pos(self, pos,l):
        return [pos.pos + l,0.0]

def _object_connections(e):
    for conn in e.findall("railml:connection",ns):
        dir = None
        if "trackBegin" in e.tag: dir = Dir.DOWN
        if "trackEnd" in e.tag: dir = Dir.UP
        if "switch" in e.tag:
            if e.attrib["orientation"] == "outgoing": dir = Dir.UP
            if e.attrib["orientation"] == "incoming": dir = Dir.DOWN
        yield (conn.attrib["id"], conn.attrib["ref"],dir)

class PointObject:
    #def find_backward(self, set=None, max_dist=None, min_dist=None):
    #    topology.find_directed(self.node.opposite(), set, max_dist, min_dist)

    def find_forward(self, set=None, max_dist=None, min_dist=None):
        self.topology.find_paths_directed(Dir.from_string(self.dir), self,
                set, max_dist, min_dist)

    def pos(self):
        return TrackRef._from_xml(self.model,self._xml_track).at_pos(float(self._xml.attrib["pos"]))

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

    def next(self, dir):
        # yield 3-tuples: (other_node, new_links, distance)
        (interval,objs) = self.topology.segments[self._segment_idx]
        next_idx = objs.index(self) + dir.factor() * 1
        if 0 <= next_idx < len(objs):
            other = objs[next_idx]
            yield (other, [], self.pos().to_pos(other.pos().pos).length)
        else:
            if dir == Dir.UP:
                for link_obj in self.topology.links[obj._segment_idx]:
                    #yield (link_obj, [], 
                    pass


class _Set(abc.Set):
    def __init__(self, l): self._items = set(l)
    def __contains__(self,item): return item in self._items
    def __iter__(self): return self._items.__iter__()
    def __len__(self): return len(self._items)

#TODO check type of objects in constructor?
class PointObjectSet(_Set):
    def __str__(self): return "Set of {} point objects.".format(len(self._items))

    def find(self,func=lambda x: True, type=None, dir=None, location=None, set=None):
        return next(iter(self.filter(func, type, dir, location, set)), None)

    def filter(self,func=lambda x: True, type=None, dir=None, location=None, set=None):
        base = self
        if type is not None:
            base = filter(lambda x: type.lower() in x._xml.tag.lower(), base)
        return PointObjectSet(filter(func, base))

    def find_forward(self, set=None, max_dist=None, min_dist=None):
        if not callable(set): set = lambda x: set
        return [path for start in self \
                     for path in start.find_forward(set(start), max_dist, min_dist)]

    def _repr_html_(self):
        return tbl(list(self._items))._repr_html_()

