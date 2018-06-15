from collections import namedtuple
from collections import defaultdict
#class InputNode(namedtuple('InputNode',["name","pos","type","links"])): pass

class Node:
    def __init__(self, type_, pos):
        self.pos = pos
        self.type = type_
        self.incoming = []
        self.outgoing = []



nodes = {
        "in":  Node("start",0.0),
        "sw0": Node("outleftsw",100.0),
        "sw1": Node("outleftsw",200.0),
        "sw2": Node("inrightsw",700.0),
        "sw3": Node("inrightsw",800.0),
        "out": Node("end",900.0) }

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


edge("in","sw0")
edge("sw0","sw3")
edge("sw0","sw1")
edge("sw1","sw2")
edge("sw1","sw2")
edge("sw2","sw3")
edge("sw3","out")


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
    def __repr__(self): return "EqCls" + str(self.idxs) + "(" + str(self.left) + "-" + str(self.right) + ")"

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
for (i,n) in enumerate(nodes.values()):
    out_ = list(map(lambda x: edges[x].eq_class, reversed(n.outgoing)))
    in_  = list(map(lambda x: edges[x].eq_class, reversed(n.incoming)))

    for ea,eb in list(zip(in_,in_[1:])) + list(zip(out_,out_[1:])):
        base.add((i,ea,eb))

def lt_relation(base):
    def overlap(edge,node):
        return edge.left <= node <= edge.right

    lt = set([(a,b) for _,a,b in base])
    delta = set(lt)
    # Semi-naive datalog
    while delta:
        up = set()
        
        # 1. transitive closue of lt relation
        up.update((a,d) for a,b in delta for _,c,d in base \
                        if b == c and not (a,d) in lt)

        # 2. using base relation as "tight" lt
        up.update((b,d) for n,a,b in base for c,d in delta \
                        if a == c and b != d and overlap(d,n) and not (b,d) in lt)
        up.update((c,a) for n,a,b in base for c,d in delta \
                        if b == d and a != c and overlap(c,n) and not (c,a) in lt)

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
edge_vars = pulp.LpVariable.dicts('y_edge', map(str,edge_classes), lowBound=0)
linprog += sum(edge_vars.values())

for ea,eb in constraints:
    linprog += edge_vars[str(ea)] + 1.0 <= edge_vars[str(eb)]

print(linprog)
print(linprog.solve())

for i,v in enumerate(linprog.variables()):
    #print(v.name, "=", v.varValue)
    edge_classes[i].level = v.varValue

for e in edge_classes:
    print ("EdgeClass " + str(e) + " \t\t@ " + str(e.level))

#y_node = {}
#for node in nodes:

for node in nodes.values():
    node.level = None
    if node.type == "start" or "in" in node.type:
        node.level = edges[node.outgoing[0]].eq_class.level
    if node.type == "end" or "out" in node.type:
        node.level = edges[node.incoming[0]].eq_class.level

for node in nodes.values():
    print("Node: " + str(node) + ", @ " + str(node.level))



linprog_x = pulp.LpProblem("x_nodes", pulp.LpMinimize)
node_vars = pulp.LpVariable.dicts('x_node', map(str, list(range(len(ordered_nodes)))), lowBound = 0)
print("NODE VARS")
print(node_vars)
linprog_x += sum(node_vars.values())

def avvik(ei,ni):
    node = ordered_nodes[ni]
    if node.type == "outleftsw" and node.outgoing[0] == ei: return True
    if node.type == "outrightsw" and node.outgoing[1] == ei: return True
    if node.type == "inleftsw" and node.incoming[1] == ei: return True
    if node.type == "inrightsw" and node.incoming[0] == ei: return True
    return False


# Connected noes have dist > 1
for (i,e) in enumerate(edges):
    dist = 1.41
    if avvik(i,e.a): dist += abs(ordered_nodes[e.a].level - e.eq_class.level)
    if avvik(i,e.b): dist += abs(ordered_nodes[e.b].level - e.eq_class.level)
    linprog_x += node_vars[str(e.a)] + dist <= node_vars[str(e.b)]

print(linprog_x)
print(linprog_x.solve())

for i,v in enumerate(linprog_x.variables()):
    #print(v.name , "=", v.varValue)
    list(nodes.values())[i].x = v.varValue

for n in ordered_nodes:
    print("Node {} @{}\t@{}".format(n.type,n.pos,n.x))


width  = max(n.x for n in ordered_nodes)
height = max(e.eq_class.level for e in edges)

import svgwrite
dwg = svgwrite.Drawing('r.svg', profile='tiny')
dwg.viewbox(-1,-1,width+2,height+2)
l = lambda a,b: dwg.add(dwg.line(a, b).stroke(color='red', width="0.01mm"))

for i,e in enumerate(edges):
    y1 = ordered_nodes[e.a].level
    y2 = e.eq_class.level
    y3 = ordered_nodes[e.b].level

    x1 = ordered_nodes[e.a].x
    x2 = x1 + abs(y2-y1)
    x4 = ordered_nodes[e.b].x
    x3 = x4 - abs(y3-y2)

    l((x1,height - y1),(x2,height - y2))
    l((x2,height - y2),(x3,height - y2))
    l((x3,height - y2),(x4,height - y3))

dwg.save()
