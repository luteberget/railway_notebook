import xml.etree.ElementTree

def read_railml(filename):
    """Read infrastructure model from railml XML file."""
    return Infrastructure(filename)



ns = {'railml': 'http://www.railml.org/schemas/2013',
        'dc': 'http://purl.org/dc/elements/1.1/'
        }

class Infrastructure:
    """Railway infrastructure model (tracks and trackside objects)"""

    def __init__(self, filename):
        """Read infrastructure model from railml XML file."""

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

        # Vis
        #self.vis = vis.vis_from_railml(tracks)

        self._xml_tracks = { (t.attrib["id"]):t for t in  tracks }
        #self.tracks = [TrackRef(self, e.attrib["id"],e.attrib["name"]) for e in tracks]

        # Topology
        #self._build_graph(

        self.switchgraph = self._build_graph(_track_switches)

    def _build_graph(self, track_objects):
        self.objects = []
        conn_id_objects = []
        for t in self._xml_tracks.values():
            objs = [Pointobject(self,t,x) for x in _sorted_pos_xml(track_objects(t))]
            self.objects += objs
            for o in objs:
                for (a,b,d) in _object_connections(o._xml):
                    conn_id_objects[a] = (b,d,o)

            for o1,o2 in zip(objs, objs[1:]):




