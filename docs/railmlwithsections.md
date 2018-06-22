# railML with sections

railML infrastructure is an object model representing information 
about rail infrastructure. More specifically, railML infrastructure
mostly describes the geometry and connectedness of railway lines 
and information about wayside equipment.

## railML infrastructure overview

The railML infrastructure model is hiearchically organized, and at the top
of the hierarchy is the **track**. (There are other top-level elements as well,
but we'll focus on tracks here because they contain most of the structure
of the railML infrastructure model). A track contains the following:

 - **Track attributes** such as main traveling direction and track type 
   (for example indicating a regularly used *main track*, or a 
    lower status *secondary track*).
 - **Topology**: 
   - A start and an end node. Each of these has a one-dimensional position,
     so the nodes together define the length of the track. Each start/end node
     may be a buffer stop or an open end, meaning that the railway simply ends
     where the track ends, or it may be a connection to another track or switch.
   - Switches, located at a distance from the start of the track, contain
     connections to the beginning or end of other tracks.
   - Crossings
 - **Track elements**:
   - **Geometric track features**, such as track radius and gradient, are
     represented by the points at which they change.
   - **Physical features**, such as tunnels, bridges, platforms, and level crossings
     are represented by a (one-dimensional) start position and a length.
   - **Functional properties** affecting trains traveling along the track, 
     such as speed restrictions, protection systems, power/electrification, 
     and service sections, are represented by a one-dimensional position where
     these properties change, and in which *direction of travel* the train is affected
     by the change. For example, speed restrictions can be dependent on the
     direction of travel.
 - **OCS elements** represent the wayside equipment used for *operational control*,
   including signals, signs, detectors, automatic train protection, balises.
   The elements have a one-dimensional position along the track, and a 
   *direction of travel*. For example, an optical signal is typically only 
   visible in one direction of travel and not the other.

Also on the top level of the hierarchy, beside the tracks, there are *lines*,
*operational control points*, and *locally controlled areas*, which 
are structures that group tracks together into larger units by cross-referencing
individual track elements.

The overall style of the model can be said to be: 

 1. Topological: locations are primarily defined as a one-dimensional distance 
    on a track, and the tracks connect to form a network.
 2. Functional: wayside elements are described by their function and not
    so much by their geometry or appearance. For example, a signal can be described
    as a distant signal or a main signal, but where its mounting equipment is connected
    to the tunnel wall is not described.

These traits make the railML infrastructure model suitable for operational 
analysis. Activities such as traveler route finding,  
scheduling of train movements, and interlocking safety analysis are
well supported by a railML infrastructure model.
Activities such as building information management, constructability analysis, 
3D visualization, and product life-cycle management are not
as well supported by representing the railway using the railML infrastructure model.

## Sections

There are some topological/functional features of railways 
which are in principle more or less covered by the railML infrastructure
object model, but require a bit of workarounds or restructuring the model in 
somewhat unnatural ways. 

 * Track attributes vs. status changes
   
 * Track grouping
   Local release areas are 

 * Detection sectoins

 * Overlapping

Splitting tracks at every possibly useful border for use in track groupings, or
for every change in track attributes, would create unnecessarily many tracks, and
for a end-user interface, this is not a good way to work.

We suggest instead to add the notion of *sections* and *section limits*:
 - **Sections** are placed at the top level of the hierarchy, outside 
   individual tracks, and define define a track vacancy detection section,
   a track siding area, local release area and similar properties of the railway
   network, without explicitly describing the section's location.
 - **Section limits** define a border for a specific section, and 
   are placed by a one-dimensional position on a track and a direction pointing
   to the interior of the section.

Sections should not be used for properties which naturally change at topological
nodes such as switches and crossings.

### Standard section types

We suggest the following sections as a standard set:

 - Track class
 - Line
 - Track vacancy detection section
 - Fouling points
 - Local release area



