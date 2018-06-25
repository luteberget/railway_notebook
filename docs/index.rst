
**spacerail** -- railway processing
===================================

**spacerail** is a Python library for computing with
railway infrastructure. It can load railML infrastructure
files and helps you build paths through the railway network,
divide the railway into sections based on delimiters, 
and perform union and intersection operation on such sections.

**spacerail** aims to help railway engineers write succinct code
for deriving information for use in interlocking tables.

Organization of code:

 * 1. NETWORK  =dgraph(nodes, edges) + Path/PathSet, (Delimiter=Node?)
 * 2. RailML = pointobjects, conv->Network
 * 3. SECTIONS
 * 4. UTILS(?) interlocking algorithms:  extended routes ??
 * 5. VISUALIZATION

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   model
   api
