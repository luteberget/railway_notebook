from collections import namedtuple
from collections import defaultdict

class Node:
    def __init__(self, type_, pos):
        self.pos = pos
        self.type = type_
        self.incoming = []
        self.outgoing = []

nodes = {
        "inb": Node("start",1000.0),
        "inb0": Node("start",1000.0),
        "outb": Node("end",2000.0),
        "t2s1": Node("outrightsw",1200.0),
        "t2s3": Node("outleftsw",1300.0),
        "t2s5": Node("outleftsw",1400.0),
        "t2s2": Node("inleftsw", 1900.0),
        "t2s4": Node("inrightsw",1800.0),
        "t2s6": Node("inrightsw",1700.0),
        "t2sz": Node("inrightsw",1500.0),

#"in": Node("start",0.0),
#"sw": Node("outleftsw", 100.0),
#"sw2": Node("outrightsw",150.0),
#"outa": Node("end",201.0),
#"outb": Node("end",151.0),
#"outc": Node("end",202.0),

#"in": Node("start", 0.0),
#"sw0": Node("outleftsw",50.0),
#"sw1": Node("outleftsw",100.0),
#"sw2": Node("inrightsw",1000.0),
#"sw3": Node("inrightsw",1200.0),
#"out": Node("end", 1300.0),

#"in": Node("start",0.0),
#"sw1": Node("outleftsw", 100.0),
#"sw3": Node("outrightsw", 200.0),
#"sw4": Node("inleftsw", 300.0),
#"sw2": Node("inrightsw", 400.0),
#"out": Node("end",500.0),



## COMPLEX
#"inx": Node("start",-1.0),
#"in": Node("start",0.0),
#"swx": Node("inleftsw",50.0),
#"sw": Node("outleftsw",100.0),
#"sw2": Node("outleftsw",150.0),
#"sw3": Node("outrightsw", 160.0),
#"out3": Node("end", 300.0),
#"outa": Node("end",300.0),
##"outb": Node("end",201.0),
#"outc": Node("end",1000.0),
#"swin": Node("inrightsw", 700.0),
#
## from swin to outc
#"lowswin":  Node("inleftsw", 900.0),
#"lowswout": Node("outleftsw", 850.0),
#"lowin": Node("start",800.0),
#"lowout": Node("end", 1000.0)


## SIMPLE CROSSOVER
#"ina": Node("start",0.0),
#"inb": Node("start",0.0),
#"sw1": Node("outleftsw",100.0),
#"sw2": Node("inleftsw",150.0),
#"outa": Node("end",200.0),
#"outb": Node("end",200.0),
        }


ordered_nodes = sorted(nodes.values(), key=lambda n: n.pos)
for i,n in enumerate(ordered_nodes): n.idx = i
edges = []

class Edge: pass
def edge(a,b):
    e = Edge()
    e.a = ordered_nodes.index(nodes[a])
    e.b = ordered_nodes.index(nodes[b])
    idx = len(edges)
    e.idx = idx
    nodes[a].outgoing.append(idx)
    nodes[b].incoming.append(idx)
    edges.append(e)

##SIMPLE CROSSOVE
#edge("ina","sw2")
#edge("sw2","outa")
#edge("inb","sw1")
#edge("sw1","sw2")
#edge("sw1","outb")

## COMPLEX
#edge("in","swx")
#edge("swx","sw")
#edge("inx","swx")
#edge("sw","sw2")
#edge("sw2","sw3")
#edge("sw3","outa")
#edge("sw3","out3")
#edge("sw2","swin")
#edge("sw","swin")
#edge("swin","lowswin")
#edge("lowswin","outc")
#edge("lowswout","lowswin")
#edge("lowin","lowswout")
#edge("lowswout", "lowout")

#edge("in","sw1")
#edge("sw1","sw3")
#edge("sw3","sw4")
#edge("sw3","sw4")
#edge("sw4","sw2")
#edge("sw1","sw2")
#edge("sw2","out")

#edge("in","sw0")
#edge("sw0","sw3")
#edge("sw0","sw1")
#edge("sw1","sw2")
#edge("sw1","sw2")
#edge("sw2","sw3")
#edge("sw3","out")

#edge("in","sw")
#edge("sw","sw2")
#edge("sw","outa")
#edge("sw2","outb")
#edge("sw2","outc")

edge("inb","t2s1")
edge("t2s1","t2s2")
edge("t2s2","outb")
edge("t2s1","t2s3")
edge("t2s3","t2s4")
edge("t2s4","t2s2")
edge("t2s3","t2s5")
edge("t2s5","t2s6")
edge("t2s5","t2sz")
edge("t2sz","t2s6")
edge("t2s6","t2s4")
edge("inb0","t2sz")



from heapq import heappush, heappop
letftof = set()

def lr_search_up(n):
    over_edges = set()
    over_nodes = set([edges[n.outgoing[0]].b])
    over = [(edges[n.outgoing[0]].b, n.outgoing[0])]
    under_edges = set()
    under_nodes = set([edges[n.outgoing[1]].b])
    under = [(edges[n.outgoing[1]].b, n.outgoing[1])]

    while over and under and not over_nodes & under_nodes:
        if over[0][0] < under[0][0]:
            _target_node,edge_idx = heappop(over)
            over_edges.add(edge_idx)
            for ei in ordered_nodes[edges[edge_idx].b].outgoing:
                over_nodes.add(edges[ei].b)
                heappush(over, (edges[ei].b, ei))
        else:
            _target_node,edge_idx = heappop(under)
            under_edges.add(edge_idx)
            for ei in ordered_nodes[edges[edge_idx].b].outgoing:
                under_nodes.add(edges[ei].b)
                heappush(under, (edges[ei].b, ei))

    over_edges.update(e for _,e in over)
    under_edges.update(e for _,e in under)
    return over_edges, under_edges

def lr_search_down(n):
    over_edges = set()
    over_nodes = set([edges[n.incoming[0]].a])
    over = [(-edges[n.incoming[0]].a, n.incoming[0])]
    under_edges = set()
    under_nodes = set([edges[n.incoming[1]].a])
    under = [(-edges[n.incoming[1]].a, n.incoming[1])]

    while over and under and not over_nodes & under_nodes:
        if over[0][0] < under[0][0]:
            _target_node,edge_idx = heappop(over)
            over_edges.add(edge_idx)
            for ei in ordered_nodes[edges[edge_idx].a].incoming:
                over_nodes.add(edges[ei].a)
                heappush(over, (-edges[ei].a, ei))
        else:
            _target_node,edge_idx = heappop(under)
            under_edges.add(edge_idx)
            for ei in ordered_nodes[edges[edge_idx].a].incoming:
                under_nodes.add(edges[ei].a)
                heappush(under, (-edges[ei].a, ei))

    over_edges.update(e for _,e in over)
    under_edges.update(e for _,e in under)
    return over_edges, under_edges

edge_lt = set()
for n in ordered_nodes:
    #print (n, n.idx, n.type)
    if "out" in n.type:
        l,r = lr_search_up(n)
        print("LEFT ",l)
        print("RIGHT ",r)
        edge_lt.update((a,b) for a in r for b in l)
    if "in" in n.type:
        l,r = lr_search_down(n)
        print("LEFT ",l)
        print("RIGHT ",r)
        edge_lt.update((a,b) for a in r for b in l)


print("LT ", edge_lt)

import pulp
linprog = pulp.LpProblem("trackplan", pulp.LpMinimize)

# Variables
# =========
#
# - node x, node y
# - edge x1,x2 (static)
#        y1,ymid,y2

M = 100.0

for n in ordered_nodes:
    n.x = pulp.LpVariable("nx" + str(n.idx), lowBound=0.0)
    n.y = pulp.LpVariable("ny" + str(n.idx), lowBound=0.0, cat="Integer")

for e in edges:
    e.y = pulp.LpVariable("ey"+str(e.idx), lowBound = 0.0, cat="Integer")
    e.dy1 = e.y - ordered_nodes[e.a].y
    e.dy2 = ordered_nodes[e.b].y - e.y
    e.isupup = pulp.LpVariable("upup{}".format(e.idx), cat="Binary")
    e.isdowndown = pulp.LpVariable("downdown{}".format(e.idx), cat="Binary")
    #e.shortup = pulp.LpVariable("shortup{}".format(e.idx), cat="Binary")
    #e.shortdown = pulp.LpVariable("shortdown{}".format(e.idx), cat="Binary")


# Constraints
# ===========

# Edge spaces node X
for e in edges:
    na = ordered_nodes[e.a]
    nb = ordered_nodes[e.b]
    dist = 1.0
    # ss-to-ss switches

    if "in" in na.type and "out" in nb.type:
        dist = 0.25

    print("DIST {} {}".format(dist,e.idx))
    linprog += ordered_nodes[e.a].x + dist <= ordered_nodes[e.b].x, "edge_dx_{}".format(e.idx)


# Node X ordering
for na,nb in zip(ordered_nodes, ordered_nodes[1:]):
    linprog += na.x <= nb.x, "node_dx0_{}_{}".format(na.idx,nb.idx)
    kmdiff = 2.0e-2 * (nb.pos - na.pos)
    linprog += na.x + kmdiff <= nb.x
    #if na.pos + 5.0 < nb.pos:
    #    linprog += na.x + 1.0 <= nb.x
    # TODO kmdiff?

def type_bools(t):
    if t == "outleftsw":  return (True,  True )
    if t == "outrightsw": return (True,  False)
    if t == "inleftsw":   return (False, True )
    if t == "inrightsw":  return (False, False)

def trunk(n):
    return edges[(n.incoming[0] if "out" in n.type else n.outgoing[0])]

def deviating_factor(n):
    out,left = type_bools(n.type)
    return (1.0 if out ^ left else -1.0)

def straight(n):
    out,left = type_bools(n.type)
    if out and left: return edges[n.outgoing[1]]
    if out and not left: return edges[n.outgoing[0]]
    if not out  and left: return edges[n.incoming[0]]
    if not out  and not left: return edges[n.incoming[1]]

def deviating(n):
    out,left = type_bools(n.type)
    if out and left: return edges[n.outgoing[0]]
    if out and not left: return edges[n.outgoing[1]]
    if not out  and left: return edges[n.incoming[1]]
    if not out  and not left: return edges[n.incoming[0]]

def end_dy(n,e):
    return (e.dy2 if "out" in n.type else e.dy1)

def start_dy(n,e):
    return (e.dy1 if "out" in n.type else e.dy2)

def set_conn_dir_start(n,e,down,straight,up):
    if "out" in n.type or "start" in n.type:
        e.e1up = up
        e.e1straight = straight
        e.e1down = down
    if "in" in n.type or "end" in n.type:
        e.e2up = up
        e.e2straight = straight
        e.e2down = down

def set_conn_dir_end(n,e,down,straight,up):
    if "out" in n.type or "start" in n.type:
        e.e2up = up
        e.e2straight = straight
        e.e2down = down
    if "in" in n.type or "end" in n.type:
        e.e1up = up
        e.e1straight = straight
        e.e1down = down

one = pulp.LpAffineExpression(constant=1.0)
zero = pulp.LpAffineExpression(constant=0.0)

# Node shape (switch, start, etc.)
for n in ordered_nodes:
    if n.type == "start":
        e = edges[n.outgoing[0]]
        linprog += e.dy1 == 0.0,"startdy1"+str(n.idx)
        set_conn_dir_start(n, e, zero, one, zero)
    elif n.type == "end":
        e = edges[n.incoming[0]]
        linprog += e.dy2 == 0.0,"enddy2"+str(n.idx)
        set_conn_dir_start(n, e, zero, one, zero)
    else:
        # Switch
        out,left = type_bools(n.type)
        #n.slanted = pulp.LpVariable("nq" + str(n.idx), cat="Binary")
        #n.slanted = zero
        #n.straight = one

        n.slanted = pulp.LpVariable("nw" + str(n.idx), cat="Binary")
        n.straight = 1-n.slanted

        
        set_conn_dir_end(n,trunk(n),
                n.slanted if left else zero,        # goes down? 
                n.straight, # goes straight?
                zero if left else n.slanted)        # goes up?

        set_conn_dir_start(n,straight(n),
                n.slanted if left else zero,        # goes down? 
                n.straight, # goes straight?
                zero if left else n.slanted)        # goes up?

        set_conn_dir_start(n,deviating(n),
                zero if left else n.straight ,              # goes down? 
                n.slanted, # goes straight? only if slanted
                n.straight if left else zero)

        # If STRAIGHT
        ##  1. trunk and straight have dy = 0 (straight connections)
        #linprog += end_dy(  n,trunk(n))    <= M*n.slanted, "trunkstraightA{}".format(n.idx)
        #linprog += end_dy(  n,trunk(n))    >= -M*n.slanted, "trunkstraightB{}".format(n.idx)
        #linprog += start_dy(n,straight(n)) <= M*n.slanted, "contstraightA{}".format(n.idx)
        #linprog += start_dy(n,straight(n)) >= -M*n.slanted, "contstraightB{}".format(n.idx)

        ##  2. deviating has dy > 0 (if upwards, which == left ??)
        #linprog += (1.0 if left else -1.0)*start_dy(n, deviating(n)) + \
        #        M*n.slanted >= 0.0, "deviating0"+str(n.idx)

        ## TODO crossover (and ladder?) need to get rid of the 1.0 part
        #linprog += (1.0 if left else -1.0)*start_dy(n, deviating(n)) + M*n.slanted >= 0.0 - M*(deviating(n).isupup + deviating(n).isdowndown)

        ##linprog += (1.0 if left else -1.0)*start_dy(n, deviating(n)) + \
        ##        M*n.slanted >= 1.0 - M*deviating(n).isupup - M*deviating(n).isdowndown,\
        ##        "deviating1"+str(n.idx)
        ##  3. node Y == trunk Y
        #linprog += n.y - trunk(n).y <= M*n.slanted, "trunknodeyA{}".format(n.idx)
        #linprog += n.y - trunk(n).y >= -M*n.slanted, "trunknodeyB{}".format(n.idx)

        ## If SLANTED
        ##  1. trunk and straight have dy < 0 (if upwards, which == left??)
        #linprog += (1.0 if left else -1.0)* end_dy(n,trunk(n))      + -M*n.straight <= -0.0
        #linprog += (1.0 if left else -1.0)* start_dy(n,straight(n)) + -M*n.straight <= -0.0
        ## TODO crossover (and ladder?) need to get rid of the 1.0 part

        #linprog += (1.0 if left else -1.0)* start_dy(n,straight(n)) + -M*n.straight <= -1.0+ M*(straight(n).isupup + straight(n).isdowndown)
        #linprog += (1.0 if left else -1.0)* end_dy(n,trunk(n))      + -M*n.straight <= -1.0 + M*(trunk(n).isupup + trunk(n).isdowndown)

        ###  2. deviating has dy = 0
        #linprog += start_dy(n, deviating(n)) <= M*n.straight
        #linprog += start_dy(n, deviating(n)) >= -M*n.straight
        ###  3. node Y == deviating Y
        #linprog += n.y - deviating(n).y <= M*n.straight
        #linprog += n.y - deviating(n).y >= -M*n.straight



# Edge shape (switches)
for e in edges:
    na = ordered_nodes[e.a]
    nb = ordered_nodes[e.b]
    absdy1,absdy2 = None,None

    print("EDGE {} {}".format(na.type,nb.type))
    if na.type == "start":
        absdy1 = e.dy1
    if na.type == "end": 
        raise Exception("Invalid model: edge left node is end node.")
    if na.type == "outleftsw" and na.outgoing[0] == e.idx: 
        # Out left deviating 
        print("Node {} {}".format(e.idx,na.type))
        absdy1 = e.dy1
    if na.type == "outleftsw" and na.outgoing[1] == e.idx: 
        # Out left straight
        absdy1 = -e.dy1
    if na.type == "outrightsw" and na.outgoing[1] == e.idx:
        # Out right deviating
        absdy1 = -e.dy1
    if na.type == "outrightsw" and na.outgoing[0] == e.idx:
        # Out right straight
        absdy1 = e.dy1
    if na.type == "inrightsw":
        absdy1 = e.dy1
    if na.type == "inleftsw":
        absdy1 = -e.dy1

    if nb.type == "start":
        raise Exception("Invalid model: edge right node is start node.")
    if nb.type == "end":
        absdy2 = e.dy2
    if nb.type == "outleftsw":
        # out left trunk
        absdy2 = -e.dy2
    if nb.type == "outrightsw":
        # out right trunk
        absdy2 = e.dy2
    if nb.type == "inrightsw" and nb.incoming[0] == e.idx:
        # in right deviating
        absdy2 = -e.dy2
    if nb.type == "inrightsw" and nb.incoming[1] == e.idx:
        # in right continuing
        absdy2 = e.dy2
    if nb.type == "inleftsw" and nb.incoming[1] == e.idx:
        # in left deviating
        absdy2 = e.dy2
    if nb.type == "inleftsw" and nb.incoming[0] == e.idx:
        # in ledt continuing
        absdy2 = -e.dy2

    e.absy = absdy1 + absdy2

    # if end is straight, dy = 0
    linprog += e.dy1 <=  M*(1-e.e1straight)
    linprog += e.dy1 >= -M*(1-e.e1straight)
    linprog += e.dy2 <=  M*(1-e.e2straight)
    linprog += e.dy2 >= -M*(1-e.e2straight)

    ## if end is down, dy <= 1.0
    linprog += e.dy1 <= 0.0 + M*(1-e.e1down)
    linprog += e.dy2 <= 0.0 + M*(1-e.e2down)
    linprog += e.dy1 + e.dy2 <= -1.0 + M*(1-e.isdowndown)
    linprog += e.dy1         <= -1.0 + M*(1-e.e1down) + M*e.isdowndown
    linprog += e.dy2         <= -1.0 + M*(1-e.e2down) + M*e.isdowndown

    ## if end is up, dy >= 1.0
    linprog += e.dy1         >= 0.0 -M*(1-e.e1up)
    linprog += e.dy2         >= 0.0 -M*(1-e.e2up)
    linprog += e.dy1 + e.dy2 >= 1.0 -M*(1-e.isupup)
    linprog += e.dy1         >= 1.0 -M*(1-e.e1up) - M*e.isupup
    linprog += e.dy2         >= 1.0 -M*(1-e.e2up) - M*e.isupup

    # if  e1up + e2up        -> dist = 0
    # if  e1down + e2down    -> dist = 0
    # if any straight or     -> dist = 1

    #linprog += na.x - nb.x + e.absy <= 0.0
    linprog += na.x + e.absy <= nb.x

    
    e.shortx   = pulp.LpVariable("shortx{}".format(e.idx), cat="Binary")
    dx = nb.x - na.x
    linprog += dx >= 2  - M*(e.shortx)
    linprog += dx <= 1  + M*(1-e.shortx)

    linprog += e.isupup >= e.e1up + e.e2up + e.shortx - 2
    linprog += e.isupup <= e.e1up
    linprog += e.isupup <= e.e2up
    linprog += e.isupup <= e.shortx
    # TODO isupup/isdowndown ONLY when dx=dy=1 ?

    linprog += e.isdowndown >= e.e1down + e.e2down + e.shortx - 2
    linprog += e.isdowndown <= e.e1down
    linprog += e.isdowndown <= e.e2down
    linprog += e.isdowndown <= e.shortx
    linprog += na.x + e.absy + 1.0 <= nb.x + M*e.isdowndown + M*e.isupup


    # e.shortup/e.shortdown if upup and na.y - nb.y + 1 >= 0 and nb.x - na.x <= 1

    #  dy = nb.y - na.y    ->  == 1,2,3,4
    #  2-dy -->  -1, 0, 1, 2, 3
    #  (dy-2) --> 1, 0, -1, -2, -3
    #dy = nb.y - na.y
    #linprog += e.isupup <= (2-dy)   #DISABLE IF not e.e1up OR not e.e2up

    # ALT:
    # if isupup: dx == 2 => dx >= 3

    # B >= C + 1 - M*(1-A);
    # C >= B + 1 - M*A
    # where B = |dx|
    # where C = 1




    pass

# Edge Y ordering
for a,b in edge_lt:
    linprog += edges[a].y <= edges[b].y, "edge_y_{}_{}".format(a,b)
    linprog += edges[a].y +1.0 <= edges[b].y + M*(edges[b].isupup + edges[a].isdowndown)


#linprog += sum(e.y for e in edges) + sum(n.x + n.y for n in ordered_nodes) + sum(e.absy for e in edges)
linprog += sum(e.y for e in edges) + sum(n.x + n.y for n in ordered_nodes) + 100*sum(e.absy for e in edges)

print(linprog)

import timeit
time_n = 20
print("t=",int(round(timeit.timeit('linprog.solve()', number=20, setup="from __main__ import linprog")/20.0*100000.0))/100.0,"ms")
print(linprog.solve())


print ("OK")

width  = max(n.x.varValue for n in ordered_nodes)#len(ordered_nodes)  #max(n.x for n in ordered_nodes)
height = max(e.y.varValue for e in edges)


if __name__ == '__main__':
    import svgwrite
    dwg = svgwrite.Drawing('t2.svg', profile='tiny')
    dwg.viewbox(-1,-1,width+2,height+2)
    c = "red"
    l = lambda a,b: dwg.add(dwg.line(a, b).stroke(color=c, width="0.01mm", opacity=1.0))
    #
    for n in ordered_nodes:
        if "sw" in n.type:
            print("Node slanted={}".format(n.slanted.varValue))
    for i,e in enumerate(edges):
    #    x1 = e.a
    #    x2 = e.b
    #    y = height - e.yvar.varValue
    #    l((x1,y),(x2,y))
        na = ordered_nodes[e.a]
        nb = ordered_nodes[e.b]

        y1 = ordered_nodes[e.a].y.varValue
        y2 = e.y.varValue
        y3 = ordered_nodes[e.b].y.varValue

        x1 = ordered_nodes[e.a].x.varValue
        x2 = x1 + abs(y2-y1)
        x4 = ordered_nodes[e.b].x.varValue
        x3 = x4 - abs(y3-y2)

        print("")
        print("edge {} {}".format(e.a,e.b),na.type,nb.type)
        print("x", x1,x2,x3,x4)
        print("y", y1,y2,y3)
        print("e1 up,straight,down", 
                e.e1up,
                e.e1straight,
                e.e1down)
        #print("yX", (zero))

        print("e2 up,straight,down", 
                e.e2up,
                e.e2straight,
                e.e2down)
        print("e.shortx", e.shortx.varValue)

        c = "black"
    #    if e.eq_class.left == 5 and e.eq_class.right == 11: c= "red"
    #    if e.eq_class.left == 2 and e.eq_class.right == 12: c= "purple"
    #    if e.eq_class.left == 8 and e.eq_class.right == 9: c="black"
        l((x1,height - y1),(x2,height - y2))
        l((x2,height - y2),(x3,height - y2))
        l((x3,height - y2),(x4,height - y3))
    #
    dwg.save()

