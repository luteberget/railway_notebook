# -*- coding: utf-8 -*-
"""rc -- railway analysis tools

This module exposes the basic building blocks for automating creation of interlocking
tables for railway control systems.
"""

import xml.etree.ElementTree
import numbers
from enum import Enum
from collections import namedtuple


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
        if not (pos_b > self.pos): raise Exception("Intervals must be given in increasing position order")
        return TrackInterval(self, pos_b - self.pos)

TrackPos.track.__doc__ = "Track reference"
TrackPos.pos.__doc__ = "Distance from start of track"

class Dir(Enum):
    UP = 1
    DOWN = 2

class Side(Enum):
    LEFT = 1
    RIGHT = 2

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
    pass


def _partition(l,pred):
    a,b = [],[]
    for x in l:
        if pred(x): a.append(x)
        else: b.append(x)
    return (a,b)

def _sorted_pos(l):
    return sorted(l, key=lambda x: float(x.attrib["pos"]))

def _track_start_conn_id(t):
    topo = t.find("railml:trackTopology",ns)
    begin = topo.find("railml:trackBegin",ns)
    return begin.attrib["id"]

def _mk_topology(model,tracks,objects):
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
        for sw in _sorted_pos(_track_switches(t)): 
            new_pos = float(sw.attrib["pos"])
            (segment_objs,objs) = _partition(objs, 
                    lambda o: pos <= float(o.attrib["pos"]) <= new_pos)
            segments.append((TrackRef._from_xml(model,t).at_pos(pos).to_pos(new_pos), _sorted_pos(segment_objs)))
            if last_segment is not None:
                links.append((last_segment,len(segments)-1))

            last_segment = len(segments)-1
            pos = new_pos
            (conn_dir,conn_id,conn_ref) = _sw_conn(sw)
            if conn_dir == Dir.UP: link_to_track.append((last_segment, conn_ref, sw))
            elif conn_dir == Dir.DOWN: start_refs[conn_id] = (last_segment,sw)

        segments.append((TrackRef._from_xml(model,t).at_pos(pos).to_pos(l), _sorted_pos(objs)))
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
            ("mileageChanges","mileageChange")
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
    for (toplevelcontainername, subcontainers) in objectelementnames:
        toplevelelement = e.find("railml:{}".format(toplevelcontainername), ns)
        if not toplevelelement: continue
        for (containername,elementname) in subcontainers:
            container = toplevelelement.find("railml:{}".format(containername), ns)
            if not container: continue
            objs += list(container.findall("railml:{}".format(elementname), ns))
    return objs

def _track_switches(e):
    topo = e.find("railml:trackTopology",ns)
    connections = topo.find("railml:connections",ns)
    sws = []
    if connections: 
        for c in connections:
            if c.tag != "{{{}}}{}".format(ns["railml"],"switch"):
                raise Exception("Connection type not supported: {}".format(c))
            sws.append(c)
    return sws


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

        # Objects
        self.objects = [PointObject(self, t, o) for t in self._xml_tracks.values() for o in _track_objects(t)]

        # Topology

    def translate_pos(self, pos,l):
        return [pos.pos + l,0.0]


class PointObject:
    def pos(self):
        return TrackRef._from_xml(self.model,self._xml_track).at_pos(float(self._xml.attrib["pos"]))

    def __getattr__(self, name):
        return self._xml.attrib[name]

    def __init__(self, model, xml_track, xml):
        self.model = model
        self._xml = xml
        self._xml_track = xml_track


