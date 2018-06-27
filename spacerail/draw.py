if __name__ == '__main__':
    from base import *
else:
    from spacerail.base import *

from collections import namedtuple
from collections import defaultdict

class Switch:
    def __init__(self, pos):
        self.pos = pos

    def straight_edge(self):
        if self.side == "left":
            return self.right
        if self.side == "right":
            return self.left

    def deviating_edge(self):
        if self.side == "left":
            return self.left
        if self.side == "right":
            return self.right

    def top_edge(self,dir):
        if dir == Dir.UP and self.dir == "outgoing":
            return self.left
        if dir == Dir.DOWN and self.dir == "incoming":
            return self.right
        raise Exception("Invalid direction for finding top edge")

    def bottom_edge(self,dir):
        if dir == Dir.UP and self.dir == "outgoing":
            return self.right
        if dir == Dir.DOWN and self.dir == "incoming":
            return self.left
        raise Exception("Invalid direction for finding bottom edge")

    def get_edges(self,dir):
        if dir == Dir.UP and self.dir == "outgoing" or \
                dir == Dir.DOWN and self.dir == "incoming":
            return [self.left, self.right]
        else:
            return [self.trunk]


class Boundary:
    def __init__(self,pos):
        self.pos = pos

    def get_edges(self,dir):
        if self.type == "start" and dir == Dir.UP: return [self.conn]
        if self.type == "end" and dir == Dir.DOWN: return [self.conn]
        return []

class Directions(namedtuple("Directions",["down","straight","up"])): pass

class Edge: 
    def __init__(self):
        self.indir = (None,None,None)
        self.outdir = (None,None,None)

    def get_node(self,dir):
        if dir == Dir.UP: 
            return self.b
        if dir == Dir.DOWN: 
            return self.a

    def dy_at(self,node):
        if self.a == node.idx:
            return self.dy1
        if self.b == node.idx:
            return self.dy2
        raise Exception()

    def absdy1(self,nodes):
        if isinstance(nodes[self.a], Switch):
            if nodes[self.a].side == "left":
                if nodes[self.a].deviating_edge() == self.idx:
                    return  self.dy1
                else:
                    return -self.dy1
            else:
                if nodes[self.a].deviating_edge() == self.idx:
                    return -self.dy1
                else:
                    return  self.dy1
        else:
            return self.dy1


    def absdy2(self,nodes):
        if isinstance(nodes[self.b], Switch):
            if nodes[self.b].side == "left":
                if nodes[self.b].deviating_edge() == self.idx:
                    return  self.dy2
                else:
                    return -self.dy2
            else:
                if nodes[self.b].deviating_edge() == self.idx:
                    return -self.dy2
                else:
                    return  self.dy2
        else:
            return self.dy2


    def set_dirs_at(self,node):
        if self.a == node.idx:
            def s(x):
                self.outdir = x
            return s
        if self.b == node.idx:
            def s(x):
                self.indir = x
            return s

class DrawTopology:
    def _parse_topo(f):
        nodes = {}
        elist = []
        for i,line in enumerate(f.split("\n")):
            l = line.split()
            if len(l) == 0 or l[0].strip().startswith("#"): continue
            cons = l.pop(0)
            if cons == "node":
                name = l.pop(0)
                type_ = l.pop(0)
                pos = float(l.pop(0))
                if len(l) > 0:
                    raise Exception("Unexpected input on line {}: {}".format(i,line))
                if "sw" in type_:
                    n = Switch(pos)
                    n.dir = "outgoing" if "out" in type_ else "incoming"
                    n.side = "left" if "left" in type_ else "right"
                    nodes[name] = n
                else:
                    n = Boundary(pos)
                    n.type = type_
                    nodes[name] = n
                
            elif cons == "edge":
                a,p = l.pop(0).split(".")
                b,q = l.pop(0).split(".")
                if len(l) > 0:
                    raise Exception("Unexpected input on line {}: {}".format(i,line))
                elist.append(((a,p),(b,q)))
            else:
                raise Exception("Unexpected input on line {}: {}".format(i,line))
        ordered_nodes = sorted(nodes.values(), key=lambda n: n.pos)
        for i,n in enumerate(ordered_nodes): n.idx = i
        edges = []
        for (a,p),(b,q) in elist:
            e = Edge()
            e.idx = len(edges)
            if nodes[a].idx > nodes[b].idx:
                a,b = b,a

            e.a = ordered_nodes.index(nodes[a])
            e.b = ordered_nodes.index(nodes[b])
            setattr(nodes[a], p, e.idx)
            setattr(nodes[b], q, e.idx)
            edges.append(e)

        return DrawTopology(ordered_nodes,edges)

    def __init__(self,nodes,edges):
        self.nodes = nodes
        self.edges = edges

    def solve(self):

        edges = self.edges
        ordered_nodes = self.nodes
        #print("edges",edges)
        #print("ordered_nodes",ordered_nodes)

        from heapq import heappush, heappop
        letftof = set()

        def lr_search(n, dir):
            over_edges = set()
            over_nodes = set([edges[n.top_edge(dir)].get_node(dir)])
            over = [(dir.factor()*edges[n.top_edge(dir)].get_node(dir), n.top_edge(dir))]
            under_edges = set()
            under_nodes = set([edges[n.bottom_edge(dir)].get_node(dir)])
            under = [(dir.factor()*edges[n.bottom_edge(dir)].get_node(dir), n.bottom_edge(dir))]

            while over and under and not over_nodes & under_nodes:
                if over[0][0] < under[0][0]:
                    _target_node, edge_idx = heappop(over)
                    over_edges.add(edge_idx)
                    for ei in ordered_nodes[edges[edge_idx].get_node(dir)].get_edges(dir):
                        over_nodes.add(edges[ei].get_node(dir))
                        heappush(over, (dir.factor()*edges[ei].get_node(dir), ei))
                else:
                    _target_node, edge_idx = heappop(under)
                    under_edges.add(edge_idx)
                    for ei in ordered_nodes[edges[edge_idx].get_node(dir)].get_edges(dir):
                        under_nodes.add(edges[ei].get_node(dir))
                        heappush(under, (dir.factor()*edges[ei].get_node(dir), ei))

            over_edges.update(e for _,e in over)
            under_edges.update(e for _,e in under)
            return over_edges, under_edges

        #def lr_search_up(n):
        #    over_edges = set()
        #    over_nodes = set([edges[n.outgoing[0]].b])
        #    over = [(edges[n.outgoing[0]].b, n.outgoing[0])]
        #    under_edges = set()
        #    under_nodes = set([edges[n.outgoing[1]].b])
        #    under = [(edges[n.outgoing[1]].b, n.outgoing[1])]

        #    while over and under and not over_nodes & under_nodes:
        #        if over[0][0] < under[0][0]:
        #            _target_node,edge_idx = heappop(over)
        #            over_edges.add(edge_idx)
        #            for ei in ordered_nodes[edges[edge_idx].b].outgoing:
        #                over_nodes.add(edges[ei].b)
        #                heappush(over, (edges[ei].b, ei))
        #        else:
        #            _target_node,edge_idx = heappop(under)
        #            under_edges.add(edge_idx)
        #            for ei in ordered_nodes[edges[edge_idx].b].outgoing:
        #                under_nodes.add(edges[ei].b)
        #                heappush(under, (edges[ei].b, ei))

        #    over_edges.update(e for _,e in over)
        #    under_edges.update(e for _,e in under)
        #    return over_edges, under_edges

        #def lr_search_down(n):
        #    over_edges = set()
        #    over_nodes = set([edges[n.incoming[0]].a])
        #    over = [(-edges[n.incoming[0]].a, n.incoming[0])]
        #    under_edges = set()
        #    under_nodes = set([edges[n.incoming[1]].a])
        #    under = [(-edges[n.incoming[1]].a, n.incoming[1])]

        #    while over and under and not over_nodes & under_nodes:
        #        if over[0][0] < under[0][0]:
        #            _target_node,edge_idx = heappop(over)
        #            over_edges.add(edge_idx)
        #            for ei in ordered_nodes[edges[edge_idx].a].incoming:
        #                over_nodes.add(edges[ei].a)
        #                heappush(over, (-edges[ei].a, ei))
        #        else:
        #            _target_node,edge_idx = heappop(under)
        #            under_edges.add(edge_idx)
        #            for ei in ordered_nodes[edges[edge_idx].a].incoming:
        #                under_nodes.add(edges[ei].a)
        #                heappush(under, (-edges[ei].a, ei))

        #    over_edges.update(e for _,e in over)
        #    under_edges.update(e for _,e in under)
        #    return over_edges, under_edges

        edge_lt = set()
        for n in ordered_nodes:
            if isinstance(n,Switch):
                dir = Dir.UP if (n.dir == "outgoing") else Dir.DOWN
                l,r = lr_search(n,dir)
                print("LEFT",l)
                print("RIGHT",r)
                edge_lt.update((a,b) for a in r for b in l)
        print("EDGE_LT",edge_lt)

        import pulp
        linprog = pulp.LpProblem("trackplan", pulp.LpMinimize)

        # Variables
        # =========
        #
        # - node x, node y
        # - edge x1,x2 (static)
        #        y1,ymid,y2

        M = 2*len(edges)

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

            if isinstance(na,Switch) and isinstance(nb,Switch) and \
                    na.dir == "incoming" and nb.dir == "outgoing":
                dist = 0.25

            #print("DIST {} {}".format(dist,e.idx))
            linprog += ordered_nodes[e.a].x + dist <= ordered_nodes[e.b].x, "edge_dx_{}".format(e.idx)


        # Node X ordering
        for na,nb in zip(ordered_nodes, ordered_nodes[1:]):
            linprog += na.x <= nb.x, "node_dx0_{}_{}".format(na.idx,nb.idx)
            kmdiff = 2.0e-2 * (nb.pos - na.pos)
            #linprog += na.x + kmdiff <= nb.x
            #if na.pos + 5.0 < nb.pos:
            #    linprog += na.x + 1.0 <= nb.x
            # TODO kmdiff?

        #def end_dy(n,e):
        #    return (e.dy2 if "out" in n.type else e.dy1)

        #def start_dy(n,e):
        #    return (e.dy1 if "out" in n.type else e.dy2)

        #def set_conn_dir_start(n,e,down,straight,up):
        #    if "out" in n.type or "start" in n.type:
        #        e.e1up = up
        #        e.e1straight = straight
        #        e.e1down = down
        #    if "in" in n.type or "end" in n.type:
        #        e.e2up = up
        #        e.e2straight = straight
        #        e.e2down = down

        #def set_conn_dir_end(n,e,down,straight,up):
        #    if "out" in n.type or "start" in n.type:
        #        e.e2up = up
        #        e.e2straight = straight
        #        e.e2down = down
        #    if "in" in n.type or "end" in n.type:
        #        e.e1up = up
        #        e.e1straight = straight
        #        e.e1down = down

        one = pulp.LpAffineExpression(constant=1.0)
        zero = pulp.LpAffineExpression(constant=0.0)

        # Node shape (switch, start, etc.)
        for n in ordered_nodes:
            if isinstance(n,Boundary):
                e = edges[n.conn]
                if n.type == "start":
                    linprog += e.dy1 == 0.0
                    e.outdir = Directions(zero, one, zero)
                if n.type == "end":
                    linprog += e.dy2 == 0.0
                    e.indir = Directions(zero, one, zero)
            elif isinstance(n,Switch):
                n.slanted = pulp.LpVariable("nw" + str(n.idx), cat="Binary")
                n.straight = 1-n.slanted

                edges[n.trunk].set_dirs_at(n)(Directions(
                        n.slanted if (n.side == "left") else zero,        # goes down? 
                        n.straight,                                       # goes straight?
                        zero if (n.side == "left") else n.slanted         # goes up?
                    ))
                edges[n.straight_edge()].set_dirs_at(n)(Directions(
                        n.slanted if (n.side == "left") else zero,        # goes down? 
                        n.straight,                                       # goes straight?
                        zero if (n.side == "left") else n.slanted         # goes up?
                    ))

                edges[n.deviating_edge()].set_dirs_at(n)(Directions(
                        zero if (n.side == "left") else n.straight,
                        n.slanted, 
                        n.straight if (n.side == "left") else zero
                        ))

        # Edge shape (switches)
        for e in edges:
            na = ordered_nodes[e.a]
            nb = ordered_nodes[e.b]
            absdy1 = e.absdy1(ordered_nodes)
            absdy2 = e.absdy2(ordered_nodes)
            e.absy = absdy1 + absdy2

            # if end is straight, dy = 0
            linprog += e.dy1 <=  M*(1-e.outdir.straight)
            linprog += e.dy1 >= -M*(1-e.outdir.straight)
            linprog += e.dy2 <=  M*(1-e.indir.straight)
            linprog += e.dy2 >= -M*(1-e.indir.straight)

            ## if end is down, dy <= 1.0
            linprog += e.dy1 <= 0.0 + M*(1-e.outdir.down)
            linprog += e.dy2 <= 0.0 + M*(1-e.indir.down)
            linprog += e.dy1 + e.dy2 <= -1.0 + M*(1-e.isdowndown)
            linprog += e.dy1         <= -1.0 + M*(1-e.outdir.down) + M*e.isdowndown
            linprog += e.dy2         <= -1.0 + M*(1-e.indir.down) + M*e.isdowndown

            ## if end is up, dy >= 1.0
            linprog += e.dy1         >= 0.0 -M*(1-e.outdir.up)
            linprog += e.dy2         >= 0.0 -M*(1-e.indir.up)
            linprog += e.dy1 + e.dy2 >= 1.0 -M*(1-e.isupup)
            linprog += e.dy1         >= 1.0 -M*(1-e.outdir.up) - M*e.isupup
            linprog += e.dy2         >= 1.0 -M*(1-e.indir.up) - M*e.isupup

            # if  e1up + e2up        -> dist = 0
            # if  e1down + e2down    -> dist = 0
            # if any straight or     -> dist = 1

            #linprog += na.x - nb.x + e.absy <= 0.0
            linprog += na.x + e.absy <= nb.x

            
            e.shortx   = pulp.LpVariable("shortx{}".format(e.idx), cat="Binary")
            dx = nb.x - na.x
            linprog += dx >= 2  - M*(e.shortx)
            linprog += dx <= 1  + M*(1-e.shortx)

            linprog += e.isupup >= e.outdir.up + e.indir.up + e.shortx - 2
            linprog += e.isupup <= e.outdir.up
            linprog += e.isupup <= e.indir.up
            linprog += e.isupup <= e.shortx
            # TODO isupup/isdowndown ONLY when dx=dy=1 ?

            linprog += e.isdowndown >= e.outdir.down + e.indir.down + e.shortx - 2
            linprog += e.isdowndown <= e.outdir.down
            linprog += e.isdowndown <= e.indir.down
            linprog += e.isdowndown <= e.shortx
            linprog += na.x + e.absy + 1.0 <= nb.x + M*e.isdowndown + M*e.isupup

        # Edge Y ordering
        for a,b in edge_lt:
            linprog += edges[a].y <= edges[b].y, "edge_y_{}_{}".format(a,b)
            linprog += edges[a].y +1.0 <= edges[b].y + M*(edges[b].isupup + edges[a].isdowndown)

        linprog += sum(e.y for e in edges) + sum(n.x + n.y for n in ordered_nodes) + 100*sum(e.absy for e in edges)


        status = linprog.solve()
        print("STATUS", status)
        if status != 1:
            raise Exception("Error in drawing topology")

        self.width  = max(n.x.varValue for n in ordered_nodes)#len(ordered_nodes)  #max(n.x for n in ordered_nodes)
        self.height = max(e.y.varValue for e in edges)

    def lines(self):
        width = self.width
        height = self.height
        ordered_nodes = self.nodes
        edges = self.edges
        for i,e in enumerate(edges):

            na = ordered_nodes[e.a]
            nb = ordered_nodes[e.b]

            y1 = ordered_nodes[e.a].y.varValue
            y2 = e.y.varValue
            y3 = ordered_nodes[e.b].y.varValue

            x1 = ordered_nodes[e.a].x.varValue
            x2 = x1 + abs(y2-y1)
            x4 = ordered_nodes[e.b].x.varValue
            x3 = x4 - abs(y3-y2)


            ls = [((x1,height - y1),(x2,height - y2)),
                  ((x2,height - y2),(x3,height - y2)),
                  ((x3,height - y2),(x4,height - y3))]
            for a,b in ls:
                if abs(b[0]-a[0]) < 1e-5 and abs(b[1]-a[1]) < 1e-5:
                    continue
                yield a,b

    def svg(self):
        width = self.width
        height = self.height
        import svgwrite
        dwg = svgwrite.Drawing(profile='tiny')
        dwg.viewbox(-1,-1,width+2,height+2)
        c = "red"
        def l(a,b):
            dwg.add(dwg.line(a, b).stroke(color=c, width="0.01mm", opacity=1.0))
        for x in self.lines(): l(*x)
        return dwg.tostring()


#nodes = {


## LADDER
#        "inb": Node("start",1000.0),
#        "inb0": Node("start",1000.0),
#        "outb": Node("end",2000.0),
#        "t2s1": Node("outrightsw",1200.0),
#        "t2s3": Node("outleftsw",1300.0),
#        "t2s5": Node("outleftsw",1400.0),
#        "t2s2": Node("inleftsw", 1850.0),
#        "t2s4": Node("inrightsw",1800.0),
#        "t2s6": Node("inrightsw",1700.0),
#        "t2sz": Node("inrightsw",1450.0),



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
        #}



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

## LADDER
#edge("inb","t2s1")
#edge("t2s1","t2s2")
#edge("t2s2","outb")
#edge("t2s1","t2s3")
#edge("t2s3","t2s4")
#edge("t2s4","t2s2")
#edge("t2s3","t2s5")
#edge("t2s5","t2s6")
#edge("t2s5","t2sz")
#edge("t2sz","t2s6")
#edge("t2s6","t2s4")
#edge("inb0","t2sz")

def draw_file(f):
    content = None
    with open(f) as fx:
        content = fx.read()
    topo = DrawTopology._parse_topo(content)
    topo.solve()
    print(topo.svg())

if __name__ == '__main__':
    import sys
    draw_file(sys.argv[1])

