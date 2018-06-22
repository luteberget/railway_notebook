.. _api:

Developer interface
===================

.. module:: spacerail


Model
-----

All 

.. autofunction:: load_railml

Some consistency checks are performed while loading, which 
can throw any of the following exceptions:

 * spacerail.InconsistentConnections
 * spacerail.UnconnectedInfrastructure


Base types
----------

.. autoclass:: Dir
    :members:
    :undoc-members:

.. autoclass:: Side
    :members:
    :undoc-members:


