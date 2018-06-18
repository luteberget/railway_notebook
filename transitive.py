from collections import namedtuple
from collections import defaultdict
#class InputNode(namedtuple('InputNode',["name","pos","type","links"])): pass

class Node:
    def __init__(self, type_, pos):
        self.pos = pos
        self.type = type_
        self.incoming = []
        self.outgoing = []


#nodes = {
#        "inv": Node("start",0.0),
#        "inh": Node("start",0.0),
#        "outv": Node("end",1000.0),
#        "outh": Node("end",1000.0),
#        "sw1": Node("outleftsw", 200.0),
#        "sw2": Node("inleftsw", 270.0),
#        }

nodes = {
        "in":  Node("start",0.0),
        "inx":  Node("start",102.0),
        "sw0": Node("outleftsw",100.0),
        "swx0": Node("outrightsw",120.0),
        "sw1": Node("outleftsw",200.0),
        "sw1y": Node("inrightsw",300.0),
        "sw1x": Node("outleftsw",300.0),
        "sw2": Node("inrightsw",700.0),
        "sw3": Node("inrightsw",800.0),
        "swx3": Node("inleftsw",820.0),
        "out": Node("end",900.0),
        "outx": Node("end",650.0),
        "inb": Node("start",1.0),
        "outb": Node("end",2000.0),
        "swb": Node("outleftsw",450.0),
        "swbx": Node("inleftsw",500.0),
        "t2s1": Node("outrightsw",1200.0),
        "t2s3": Node("outleftsw",1300.0),
        "t2s5": Node("outleftsw",1400.0),
        "t2s2": Node("inleftsw", 1900.0),
        "t2s4": Node("inrightsw",1800.0),
        "t2s6": Node("inrightsw",1700.0),
        }

ordered_nodes = sorted(nodes.values(), key=lambda n: n.pos)
edges = []

class Edge: pass
def edge(a,b):
    e = Edge()
    e.a = ordered_nodes.index(nodes[a])
    e.b = ordered_nodes.index(nodes[b])
    idx = len(edges)
    nodes[a].outgoing.append(idx)
    nodes[b].incoming.append(idx)
    edges.append(e)

#edge("inv","sw2")
#edge("sw2","outv")
#edge("sw1","sw2")
#edge("inh","sw1")
#edge("sw1","outh")

edge("inx","sw1y")
edge("sw1x","outx")
edge("sw1x","sw2")
edge("in","sw0")
edge("sw0","sw3")
edge("sw0","swx0")
edge("swx0","sw1")
edge("sw1","sw1y")
edge("sw1y","sw1x")
edge("sw1","sw2")
edge("sw2","sw3")
edge("sw3","swx3")
edge("swx3","out")
edge("swx0","swbx")
edge("swbx", "swx3")
edge("inb","swb")
edge("swb","swbx")
edge("swb","t2s1")
edge("t2s1","t2s2")
edge("t2s2","outb")
edge("t2s1","t2s3")
edge("t2s3","t2s4")
edge("t2s4","t2s2")
edge("t2s3","t2s5")
edge("t2s5","t2s6")
edge("t2s5","t2s6")
edge("t2s6","t2s4")


def eq_classes(items, relations):
    id = list(range(len(items)))

    def root(idx):
        while idx != id[idx]: 
            id[idx] = id[id[idx]]
            idx = id[idx]
        return idx

    for a,b in relations:
        a,b = root(a),root(b)
        id[a] = b

    d = defaultdict(list)
    for i,n in enumerate(items):
        d[root(i)].append(n)
    return list(d.values())

print("all edges:")
print(edges)

class EdgeEqClass:
    def __init__(self, idxs):
        self.left  = min(edges[i].a for i in idxs)
        self.right = max(edges[i].b for i in idxs)
        self.idxs = idxs
        for i in idxs: edges[i].eq_class = self
    def __repr__(self): return "EqCls"  + "(" + str(self.left) + "-" + str(self.right) + ")"

eqs = []
for node in nodes.values():
    if node.type == "outleftsw":
        eqs.append((node.incoming[0], node.outgoing[1]))
    if node.type == "outrightsw":
        eqs.append((node.incoming[0], node.outgoing[0]))
    if node.type == "inleftsw":
        eqs.append((node.incoming[0], node.outgoing[0]))
    if node.type == "inrightsw":
        eqs.append((node.incoming[1], node.outgoing[0]))

print("EQS")
print(eqs)

#print(eq_classes(list(range(len(edges))),eqs))

edge_classes = [EdgeEqClass(a) for a in eq_classes(list(range(len(edges))), eqs)]
print("Edge classes")
print(edge_classes)


base = set()
for (i,n) in enumerate(ordered_nodes): #enumerate(nodes.values()):
    out_ = list(map(lambda x: edges[x].eq_class, reversed(n.outgoing)))
    in_  = list(map(lambda x: edges[x].eq_class, reversed(n.incoming)))

    for ea,eb in list(zip(in_,in_[1:])) + list(zip(out_,out_[1:])):
        base.add((i,ea,eb))

def lt_relation(base):
    def overlap(edge,node):
        return edge.left < node < edge.right

    lt = set([(a,b) for _,a,b in base])
    delta = set(lt)
    # Semi-naive datalog
    while delta:
        up = set()
        
        # 1. transitive closue of lt relation
        up0 = list((a,d) for a,b in delta for _,c,d in base \
                        if b is c and not (a,d) in lt)
        print("Update 0",up0)
        up.update(up0)

        # 2. using base relation as "tight" lt
        up1 = list((b,d) for n,a,b in base for c,d in delta \
                        if a is c and b is not d and overlap(d,n) and not (b,d) in lt)

        print("Update 1",up1)
        up.update(up1)

        up2 = list((c,a) for n,a,b in base for c,d in delta \
                        if b is d and a is not c and overlap(c,n) and not (c,a) in lt)

        print("Update 2",up2)
        up.update(up2)

        # TODO these updates are using a cartesian join, indexing the relations
        # by columns could allow for hash join or merge join

        lt.update(up)
        delta = up
    return lt

print("BASE")
print(base)
print("LT RELATION")
print(lt_relation(base))

lt = lt_relation(base)

# Create constraints based on lt relation

#nodes = [0,0,0]
#edges = set(x for _,a,b in base for x in (a,b))
#edge_vars = dict(enumerate(edges))
constraints = set()

from functools import cmp_to_key
for (i,n) in enumerate(nodes):
    node_edges = [ec for ec in edge_classes if ec.left <= i <= ec.right]
    node_edges.sort(key= cmp_to_key(lambda x,y: -1 if (x,y) in lt else 1))
    constraints.update((a,b) for a,b in zip(node_edges, node_edges[1:]))
 
print("CONSSTRAINTS", len(constraints), constraints)

#def estring(e): return str(e.a) +"_"+ str(e.b)
import pulp
linprog = pulp.LpProblem("y_edges", pulp.LpMinimize)
#edge_vars = pulp.LpVariable.dicts('y_edge', map(str,edge_classes), lowBound=0)
for ec in edge_classes: ec.yvar = pulp.LpVariable("y_edge" + str(ec), lowBound=0)
for i,n in enumerate(ordered_nodes):
    n.xvar = pulp.LpVariable("x_node_" + str(i), lowBound=0.0)

for node in nodes.values():
    node.level = None
    if node.type == "start" or "in" in node.type:
        print(node.type)
        node.levelvar = edges[node.outgoing[0]].eq_class.yvar
    if node.type == "end" or "out" in node.type:
        node.levelvar = edges[node.incoming[0]].eq_class.yvar


def sloeyfe(e):
    if len(e.idxs) != 1: return False
    na = ordered_nodes[edges[e.idxs[0]].a]
    nb = ordered_nodes[edges[e.idxs[0]].b]
    leftcrossover  = na.type == "outleftsw" and nb.type == "inleftsw"
    rightcrossover = na.type == "outrightsw" and nb.type == "inrightsw"
    return leftcrossover or rightcrossover

sloeyfe_vars = []
for ea,eb in constraints:
    if sloeyfe(eb): 
        # EB er en sløyfe som knytter sammen EA og EX
        # gjennom node OVER og UNDER
        after    = ordered_nodes[edges[eb.idxs[0]].b]
        before   = ordered_nodes[edges[eb.idxs[0]].a]
        
        # ea.y + x <= eb.y
        # hvor X er min(1.0, dx)
        ny_x = pulp.LpVariable("sloeyfe_" + str(edges[eb.idxs[0]].a) + "_" + str(edges[eb.idxs[0]].b), lowBound=0.0, upBound=1.0)
        dx = after.xvar - before.xvar - 1.0 # in [0,->
        linprog += ny_x <= dx, "slX"
        linprog += ea.yvar + ny_x <= eb.yvar
        sloeyfe_vars.append(ny_x)
    else: linprog += ea.yvar + 1.0 <= eb.yvar
    #pass

#print(linprog)
#print(linprog.solve())

#for ec in edge_classes:
#    ec.level = ec.yvar.varValue
#for i,v in enumerate(edge_vars.values()):
    #print(v.name, "=", v.varValue)
    #dge_classes[i].level = v.varValue
    #for ec in edge_classes:
    #    if v.name == "y_edge_" + str(ec):
    #        ec.level = v.varValue

#for e in edge_classes:
#    print ("EdgeClass " + str(e) + " \t\t@ " + str(e.level))

#y_node = {}
#for node in nodes:


#for node in nodes.values():
#    print("Node: " + str(node) + ", @ " + str(node.level))



#linprog_x = pulp.LpProblem("x_nodes", pulp.LpMinimize)
#node_vars = pulp.LpVariable.dicts('x_node', map(str, list(range(len(ordered_nodes)))), lowBound = 0)
#print("NODE VARS")
#print(node_vars)


def avvik(ei,ni):
    node = ordered_nodes[ni]
    if node.type == "outleftsw" and node.outgoing[0] == ei: return True
    if node.type == "outrightsw" and node.outgoing[1] == ei: return True
    if node.type == "inleftsw" and node.incoming[1] == ei: return True
    if node.type == "inrightsw" and node.incoming[0] == ei: return True
    return False


# Connected noes have dist > 1
# TODO: unless they are facing facing-facing switches? 
for (i,e) in enumerate(edges):
    dist = 1.0

    # unless crossover
    if ordered_nodes[e.a].type == "outleftsw" and ordered_nodes[e.b].type == "inleftsw": dist = 0.0
    if ordered_nodes[e.a].type == "outrightsw" and ordered_nodes[e.b].type == "inrightsw": dist = 0.0

    if avvik(i,e.a): 
        if ordered_nodes[e.a].outgoing[1] == i: 
            #downward
            dist += ordered_nodes[e.a].levelvar - e.eq_class.yvar
        else:
            #upward
            dist += e.eq_class.yvar - ordered_nodes[e.a].levelvar

        #if "out" in ordered_nodes[e.a].type and 
        #dist += abs(ordered_nodes[e.a].level - e.eq_class.level)
    if avvik(i,e.b): 
        if ordered_nodes[e.b].incoming[1] == i:
            #downward
            dist += ordered_nodes[e.b].levelvar - e.eq_class.yvar
        else:
            #upward
            dist += e.eq_class.yvar - ordered_nodes[e.b].levelvar
        #dist += abs(ordered_nodes[e.b].level - e.eq_class.level)
    linprog += ordered_nodes[e.a].xvar + dist <= ordered_nodes[e.b].xvar

# Km ordering (non-strictly) increasing
for i in range(len(ordered_nodes)-1):
    mindist = 0.25
    linprog += ordered_nodes[i].xvar + mindist <= ordered_nodes[i+1].xvar

    # TODO maximum length distortion constraint?
    kmdiff = 0.5e-2 * (ordered_nodes[i+1].pos - ordered_nodes[i].pos)
    #linprog += ordered_nodes[i].xvar + kmdiff <= ordered_nodes[i+1].xvar
    pass

# If start/end point inside the slanted area
for (i,e) in enumerate(edges):
    if avvik(i,e.a):
        # Starts in avvik, make sure that any start nodes inside the edge have space
        # any start nodes inside must be at least H away
        for j,n in enumerate(ordered_nodes):
            if n.type == "start" and e.a < j < e.b:
                o = edges[[x for x in ordered_nodes[e.a].outgoing if x != i][0]].eq_class
                nin = edges[n.outgoing[0]].eq_class
                dis = e.eq_class
                mellom = ((o,nin) in lt and (nin,dis) in lt) or ((dis,nin) in lt and (nin,o) in lt)
                if mellom:
                    h = n.levelvar - ordered_nodes[e.a].levelvar
                    if ordered_nodes[e.a].outgoing[1] == i: h *= -1.0 #downward
                    linprog += ordered_nodes[e.a].xvar + h + 1.5 <= n.xvar
    if avvik(i,e.b):
        for j,n in enumerate(ordered_nodes):
            if n.type == "end" and e.a < j < e.b:
                o = edges[[x for x in ordered_nodes[e.b].incoming if x != i][0]].eq_class
                nin = edges[n.incoming[0]].eq_class
                dis = e.eq_class
                mellom = ((o,nin) in lt and (nin,dis) in lt) or ((dis,nin) in lt and (nin,o) in lt)
                if mellom:
                    print("MELLOM")
                    h = n.levelvar - ordered_nodes[e.b].levelvar
                    if ordered_nodes[e.b].incoming[1] == i: h *= -1.0 #downward
                    #h = abs(ordered_nodes[e.b].level - n.level) + 1.0 #e.eq_class.level)
                    print("DOWN AVVIK END")
                    print(h)
                    linprog += n.xvar + h + 1.5 <= ordered_nodes[e.b].xvar
                else:
                    print("NOT MELLOM")

# TODO what if a switch connects inside the slanted area?
# answer: it doesn't, because there is enough room on the same edge...

linprog += sum(ec.yvar for ec in edge_classes) + 100.0*sum(n.xvar for n in ordered_nodes) + sum(-10.0*x for x in sloeyfe_vars)
print(linprog)
print(linprog.solve())

#for i,v in enumerate(linprog_x.variables()):
    #print(v.name , "=", v.varValue)
    #list(nodes.values())[i].x = v.varValue
for n in ordered_nodes:
    n.x = n.xvar.varValue
    n.level = n.levelvar.varValue

for ec in edge_classes:
    ec.level = ec.yvar.varValue


for n in ordered_nodes:
    print("Node {} @{}\t@{} \t@{}".format(n.type,n.pos,n.x,n.level))
for ec in edge_classes:
    print("Edge class {} \t\t@{}".format(str(ec),ec.level))
for s in sloeyfe_vars:
    print("Sloeyfe NYX {}".format(s.varValue))


width  = max(n.x for n in ordered_nodes)
height = max(e.eq_class.level for e in edges)

import svgwrite
dwg = svgwrite.Drawing('r.svg', profile='tiny')
dwg.viewbox(-1,-1,width+2,height+2)
c = "red"
l = lambda a,b: dwg.add(dwg.line(a, b).stroke(color=c, width="0.01mm", opacity=0.5))

for i,e in enumerate(edges):
    y1 = ordered_nodes[e.a].level
    y2 = e.eq_class.level
    y3 = ordered_nodes[e.b].level

    x1 = ordered_nodes[e.a].x
    x2 = x1 + abs(y2-y1)
    x4 = ordered_nodes[e.b].x
    x3 = x4 - abs(y3-y2)

    c = "blue"
    if e.eq_class.left == 5 and e.eq_class.right == 11: c= "red"
    if e.eq_class.left == 2 and e.eq_class.right == 12: c= "purple"
    if e.eq_class.left == 8 and e.eq_class.right == 9: c="black"
    l((x1,height - y1),(x2,height - y2))
    l((x2,height - y2),(x3,height - y2))
    l((x3,height - y2),(x4,height - y3))

dwg.save()
