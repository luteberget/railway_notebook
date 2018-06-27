# function to import railml from file

import xml.etree.ElementTree
from spacerail.network import *
from spacerail.base import *
from spacerail import draw
from math import inf

ns = {'railml': 'http://www.railml.org/schemas/2013',
                'dc': 'http://purl.org/dc/elements/1.1/'}

switchelementnames = [ ("trackTopology", [("connections","switch")]) ]

objectelementnames = [
        ("trackTopology", [
            ("connections","switch"),
            ("connections","crossing"),
            ("mileageChanges","mileageChange"),
            ]),
        ("trackElements", [
            ("radiusChanges","radiusChange"),
            ("gradientChanges","gradientChange"),
            ("trackConditions","trackCondition"),
            ]),
        ("ocsElements", [
            ("signals","signal"),
            ("trainDetectionElements","trainDetector"),
            ("derailers","derailer"),
            ]),
        ]


def mk_vis_model(tracks):

    class Continuation(namedtuple("Continuation",["a","b"])): pass
    class Switch(namedtuple("Switch",["side","trunk","left","right"])): pass

    class Port:
        def __init__(self):
            self.other = None
            self.conn = None
            self.node = None

        def port_name(self):
            if isinstance(self.node, Continuation): return "conn"
            if isinstance(self.node, Switch):
                if self.node.trunk is self: return "trunk"
                if self.node.left is self: return "left"
                if self.node.right is self: return "right"

    named_ports = {}
    start = None
    nodes = []

    for t in tracks:
        #print(t.attrib["name"])
        node_elements = _sorted_pos_xml(_track_objects(t,switchelementnames))
        last_pos = 0.0
        prev_port = None
        for el in node_elements:
            pos = float(el.attrib["pos"])
            if "switch" in el.tag:
                node = Switch(_switch_connection_course(el), Port(), Port(), Port())
                node.id = el.attrib["id"]
                node.trunk.other = [node.left, node.right]
                node.left.other = [node.trunk]
                node.right.other = [node.trunk]
                node.trunk.node = node
                node.left.node = node
                node.right.node = node
                #print("Switch node {}".format(len(nodes)))
                nodes.append(node)

                if _switch_orientation(el) == Dir.UP:
                    prev_port.conn = (node.trunk, pos-last_pos)
                    node.trunk.conn = (prev_port, pos-last_pos)
                    #print("conncting trunk",prev_port.node,node)
                    if _switch_connection_course(el) == Side.LEFT:
                        #print("left")
                        prev_port = node.right
                    else:
                        #print("right")
                        prev_port = node.left
                else:
                    if _switch_connection_course(el) == Side.LEFT:
                        prev_port.conn = (node.right, pos-last_pos)
                        node.right.conn = (prev_port, pos-last_pos)
                        #print("conncting right",prev_port.node,node)
                    else:
                        prev_port.conn = (node.left, pos-last_pos)
                        node.left.conn = (prev_port, pos-last_pos)
                        #print("conncting left",prev_port.node,node)
                    prev_port = node.trunk

                conn = _object_connection(el)
                if conn:
                    #print ("ADDING SWITCH CONN", conn)
                    if _switch_connection_course(el) == Side.LEFT:
                        named_ports[conn[0]] = (conn[1], node.left)
                    else:
                        named_ports[conn[0]] = (conn[1], node.right)
                else:
                    print("Warning: no connection in switch ", el.attrib["id"])

            else:
                node = Continuation(Port(), Port())
                node.id = t.attrib["id"] + el.tag
                node.a.other = [node.b]
                node.b.other = [node.a]
                node.a.node = node
                node.b.node = node
                nodes.append(node)
                #print("Cont node {}".format(len(nodes)))

                if not prev_port: # track begin
                    conn = _object_connection(el)
                    if conn:
                        #print ("ADDING BEGIN CONN", conn)
                        named_ports[conn[0]] = (conn[1], node.a)
                    else:
                        if not start:
                            start = node.b
                else:
                    prev_port.conn = (node.a, pos-last_pos)
                    node.a.conn = (prev_port, pos-last_pos)
                    #print("conncting cont",prev_port.node,node)
                    conn = _object_connection(el)
                    if conn:
                        #print ("ADDING END CONN", conn)
                        named_ports[conn[0]] = (conn[1], node.b)

                prev_port = node.b
            last_pos = pos

    while named_ports:
        id1,(ref1,port_a) = named_ports.popitem()
        ref2,port_b       = named_ports.pop(ref1)
        if not id1 == ref2: raise Exception("Inconsistent connections in railML file.")
        #print("RESOLVING", id1, ref1, ref2)
        # Connect them
        port_a.conn = (port_b, 0.0)
        port_b.conn = (port_a, 0.0)
        #print("conncting cont",port_a.node,port_b.node)

    def removecont(n):
        if isinstance(n,Continuation) and (n.a.conn is not None) and (n.b.conn is not None):
            a,l1 = n.a.conn
            b,l2 = n.b.conn
            a.conn = (b, l1+l2)
            b.conn = (a, l1+l2)
            return True 
        return False 
    nodes = [n for n in nodes if not removecont(n)]
    #print("nodes: {}".format(len(nodes)))
    #print(start)

    start.node.type = "start"
    start.node.abspos = 0.0
    stack = [(start.conn[0],1,start.conn[1])]
    visited = set([start.node])
    while stack:
        port,dir,pos = stack.pop()
        #print("POP")
        #print(port.node,dir,port)
        #print("set abspos", id(port.node), pos)
        port.node.abspos = pos
        if isinstance(port.node, Continuation):
            if dir < 0: port.node.type = "start"
            if dir > 0: port.node.type = "end"

        if isinstance(port.node, Switch):
            if len(port.other) == 2: # outgoing
                if dir > 0: port.node.type = "outgoing"
                if dir < 0: port.node.type = "incoming"
            else:
                if dir < 0: port.node.type = "outgoing"
                if dir > 0: port.node.type = "incoming"

        for x in port.other:
            #print("PORT.OTHER",x.conn)
            if x.conn is None:
                continue
            if not x.conn[0].node in visited:
                visited.add(x.conn[0].node)
                stack.append((x.conn[0],dir,pos+dir*x.conn[1]))
            for y in x.other:
                if y.conn is None:
                    continue
                if not y.conn[0].node in visited:
                    visited.add(y.conn[0].node)
                    stack.append((y.conn[0],-dir,pos+(-dir)*y.conn[1]))


    edges = []
    named_nodes = {}
    for i,n in enumerate(nodes): n.name = i
    for n in nodes:
        if isinstance(n,Continuation):
            dn = draw.Boundary(n.abspos)
            dn.type = n.type
            named_nodes[n.name] = dn
            if n.a.conn is not None:
                edges.append(((n.name,"conn"),(n.a.conn[0].node.name,n.a.conn[0].port_name())))
            if n.b.conn is not None:
                edges.append(((n.name,"conn"),(n.b.conn[0].node.name,n.b.conn[0].port_name())))
        if isinstance(n, Switch):
            dn = draw.Switch(n.abspos)
            dn.dir = n.type
            dn.side = "left" if n.side == Side.LEFT else "right"
            #print("SWITCH",dn.dir,dn.side)
            named_nodes[n.name] = dn
            edges.append(((n.name,"trunk"),(n.trunk.conn[0].node.name, n.trunk.conn[0].port_name())))
            edges.append(((n.name,"left"),(n.left.conn[0].node.name, n.left.conn[0].port_name())))
            edges.append(((n.name,"right"),(n.right.conn[0].node.name, n.right.conn[0].port_name())))

    ordered_nodes = sorted(named_nodes.values(), key = lambda x: x.pos)
    for i,n in enumerate(ordered_nodes): n.idx = i

    draw_edges = []
    for (an,ap),(bn,bp) in edges:
        if named_nodes[an].idx > named_nodes[bn].idx: continue
        e = draw.Edge()
        e.idx = len(draw_edges)
        e.a = named_nodes[an].idx
        e.b = named_nodes[bn].idx
        setattr(named_nodes[an], ap, e.idx)
        setattr(named_nodes[bn], bp, e.idx)
        draw_edges.append(e)

    #for e in draw_edges:
        #print("edge", e.a,e.b)

    topo = draw.DrawTopology(ordered_nodes, draw_edges)
    topo.solve()
    return topo



def _object_connections(e):
    for conn in e.findall("railml:connection",ns):
        dir = None
        if "trackBegin" in e.tag: dir = Dir.DOWN
        if "trackEnd" in e.tag: dir = Dir.UP
        if "switch" in e.tag: dir = _switch_orientation(e)
        yield (conn.attrib["id"], conn.attrib["ref"],dir)

def _object_connection(e):
    for conn in e.findall("railml:connection",ns):
        dir = None
        if "trackBegin" in e.tag: dir = Dir.DOWN
        if "trackEnd" in e.tag: dir = Dir.UP
        if "switch" in e.tag: dir = _switch_orientation(e)
        return (conn.attrib["id"], conn.attrib["ref"],dir)

class Infrastructure:
    def _repr_svg_(self):
        return self.vis.svg()

    #@property
    #def objects(self):
    #    return (obj for node in self.network.nodes for obj in node.objects)

def read_railml(filename):
    """Import a railML file containing infrastructure.

    :param filename: File name of railML file
    :rtype: Network
    """

    # Load infrastructure from file
    tree = xml.etree.ElementTree.parse(filename).getroot()
    iss = list(tree.findall('railml:infrastructure',ns))
    if len(iss) != 1: raise Exception("Infrastructure not found in file.")
    infrastructure = iss[0]

    # Tracks
    tracks = list(infrastructure.findall('railml:tracks',ns))
    if len(tracks) != 1: raise Exception("Infrastructure contains no tracks.")
    tracks = tracks[0]
    tracks = list(tracks.findall('railml:track',ns))
    if len(tracks) == 0: raise Exception("Infrastructure contains no tracks.")


    i = Infrastructure()
    i.vis = mk_vis_model(tracks)
    i.network = mk_network_model(tracks)
    i.objects = PointObjectSet(obj for node in i.network.nodes for obj in node.objects)
    for o in i.objects: o._inf = i
    return i


def mk_network_model(tracks):
    named_connections = {}
    network = Network()

    for t in tracks:
        objs = (PointObject(t,x) for x in _sorted_pos_xml(_track_objects(t,objectelementnames)))
        last_pos = 0.0
        last_node = None
        for obj in objs:
            na,nb = network.mk_node_pair()

            for elem,node in [("trackBegin",na),("trackEnd",nb)]:
                if elem in obj.type:
                    for conn in obj._xml.findall("railml:connection",ns):
                        named_connections[conn.attrib["id"]] = (conn.attrib["ref"], node)

            if "switch" in obj.type:
                new_na_conn,new_nb_conn = network.mk_node_pair()
                new_na_cont,new_nb_cont = network.mk_node_pair()
                node = nb if obj.dir == Dir.UP else na

                conn_side = _switch_connection_course(obj._xml)
                node.edges.append(       EdgeData(node,new_na_conn, 0.0, (obj, conn_side)))
                new_na_conn.edges.append(EdgeData(new_na_conn,node, 0.0, (obj, conn_side)))

                cont_side = conn_side.opposite()
                nb.edges.append(         EdgeData(nb,new_na_cont, 0.0, (obj, cont_side)))
                new_na_cont.edges.append(EdgeData(new_na_cont,nb, 0.0, (obj, cont_side)))

                for conn in obj._xml.findall("railml:connection",ns):
                    named_connections[conn.attrib["id"]] = (conn.attrib["ref"], new_nb_conn)

                if obj.dir == Dir.UP:
                    nb = new_nb_cont
                else:
                    na = new_nb_cont

            if last_node is not None:
                na.edges.append(       EdgeData(na,last_node,obj.pos-last_pos,None))
                last_node.edges.append(EdgeData(last_node,na,obj.pos-last_pos,None))


            if obj.dir == Dir.UP:
                nb.objects.append(obj)
                obj.node = nb
            else:
                # If dir is unknown/both/error, then put it in node A.
                na.objects.append(obj)
                obj.node = na

            last_pos = obj.pos
            last_node = nb

    while named_connections:
        id1,(ref1,node_a) = named_connections.popitem()
        ref2,node_b       = named_connections.pop(ref1)
        if not id1 == ref2: raise Exception("Inconsistent connections in railML file.")
        node_a.edges.append(EdgeData(node_a,node_b,0.0,None))
        node_b.edges.append(EdgeData(node_b,node_a,0.0,None))

    return network
    
def _sorted_pos_xml(l):
        return sorted(l, key=lambda x: float(x.attrib["pos"]))

def _track_objects(e, names):
    objs = []
    topo = e.find("railml:trackTopology",ns)
    objs.append(topo.find("railml:trackBegin",ns))
    objs.append(topo.find("railml:trackEnd",ns))
    for (toplevelcontainername, subcontainers) in names:
        toplevelelement = e.find("railml:{}".format(toplevelcontainername), ns)
        if not toplevelelement: continue
        for (containername,elementname) in subcontainers:
            container = toplevelelement.find("railml:{}".format(containername), ns)
            if not container: continue
            objs += list(container.findall("railml:{}".format(elementname), ns))
    return objs


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
    def __init__(self, xml_track, xml):
        self._xml_track = xml_track
        self._xml = xml

    def getproperty(self,name):
        try:
            return self._xml.attrib[name]
        except KeyError:
            return None

    @property
    def type(self):
        return self._xml.tag

    @property
    def pos(self):
        return float(self._xml.attrib["pos"])

    @property
    def dir(self):
        try:
            return Dir.from_string(self._xml.attrib["dir"])
        except KeyError:
            return None

    @property
    def id(self):
        return self.getproperty("id")

    def find_forward(self, goals, max_dist=inf, min_dist=0.0, dir=None):
        if dir is not None:
            is_goal = (lambda x, same: (x in goals and same == dir))
        else:
            is_goal = (lambda x, same: (x in goals))

        return self._inf.network.search(self.node, is_goal, max_dist, min_dist)

import pandas
class PointObjectSet(ExtendSet):
    def __str__(self): return "Set of {} point objects.".format(len(self._items))

    def filter(self, func=None, type=None, dir=None, location=None, id=None):
        if func is None: func = lambda x: True
        base = self
        if type is not None:
            base = filter(lambda x: type.lower() in x._xml.tag.lower(), base)
        if id is not None:
            base = filter(lambda x: id == x.id, base)
        if location is not None:
            base = filter(lambda x: location.contains(x), base)
        if dir is not None:
            base = filter(lambda x: x.dir == dir, base)
        return PointObjectSet(filter(func, base))

    def table(self,fields=None):
        if fields is None: fields = ["id","code","name","dir","pos"]
        return pandas.DataFrame.from_records({k:x.getproperty(k) for k in fields} for x in self)

    def find_forward(self, goals, max_dist=inf, min_dist=0.0, dir=None):
        """Find objects forward"""
        goals_f = goals if callable(goals) else (lambda x: goals)
        return frozenset(path for start in self \
                              for path in start.find_forward(goals_f(start), max_dist, min_dist, dir))

    def _repr_html_(self):
        return self.dataframe()._repr_html_()


