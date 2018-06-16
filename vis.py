# -*- coding: utf-8 -*-

class Node:
    def __init__(self, type_, pos):
        self.pos = pos
        self.type = type_
        self.incoming = []
        self.outgoing = []

def vis_from_railml(xml_tracks, use_abspos=None):
    print("vis")
    if use_abspos is None: use_abspos = True

    # 1. Create a covering mapping from TrackIntervals to lines 
    # (this also gives an x coordinate, which can be scaled up to
    #  the average of the lengths of tracks between consecutive nodes)
    #

    for t in xml_tracks:
        for t in swtic
        if t.begin is Connection:
            

    pass
