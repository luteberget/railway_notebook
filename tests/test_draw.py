import unittest
import pickle

import sys
sys.path.append('..')
import spacerail.draw

class TestTopologyOutput(unittest.TestCase):

    def test_files(self):
        from os.path import isfile, join, dirname, abspath
        from os import listdir
        path = "./draw_cases"
        path = join(dirname(abspath(__file__)), path)
        tests = [f for f in listdir(path) if isfile(join(path, f)) and \
                f.endswith(".topo")]
        for x in tests: 
            with self.subTest(filename=x):
                self.dofile(join(path,x))


    def dofile(self, f):
        def _round(x):
            return (int(round(10.0*x[0])), int(round(10.0*x[1])))

        with open(f) as fx: 
            contents = fx.read()
        topo = spacerail.draw.DrawTopology._parse_topo(contents)
        topo.solve()
        lines = topo.lines()
        lineset = frozenset( (_round(a),_round(b)) for a,b in lines)
        print(lineset)

        with open(f+".output.pickle","wb") as fx:
            pickle.dump(lineset,fx)

        with open(f+".output.svg","w") as fx:
            fx.write(topo.svg())

        with open(f+".correct.pickle", 'rb') as fx:
            correct_lineset = pickle.load(fx)
            assert(lineset == correct_lineset)

if __name__ == '__main__':
    unittest.main()
