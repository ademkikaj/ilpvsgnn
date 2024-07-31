:- use_module(library(aleph)).
:- if(current_predicate(use_rendering/1)).
:- use_rendering(prolog).
:- endif.
:- aleph.
:- style_check(-discontiguous).
:- aleph_set(verbosity,1).
:- aleph_set(minpos,5).
:- aleph_set(nodes,100000).
:- modeh(1,ptc(+id)).
:- modeb(*,atom(+id, -node_id, -atom_type)).
:- modeb(*,connected(+id,-node_id,-node_id,-bond_type)).
:- modeb(*,eq_atom(+atom_type, -atom_type)).
:- modeb(*,eq_bond(+bond_type, -bond_type)).
:- modeb(*,cycle3(+id, -node_id, -node_id, -node_id)).
:- modeb(*,cycle4(+id, -node_id, -node_id, -node_id, -node_id)).
:- modeb(*,cycle5(+id, -node_id, -node_id, -node_id, -node_id, -node_id)).
:- determination(ptc/1,atom/3).
:- determination(ptc/1,connected/4).
:- determination(ptc/1,eq_atom/2).
:- determination(ptc/1,eq_bond/2).
:- determination(ptc/1,cycle3/4).
:- determination(ptc/1,cycle4/5).
:- determination(ptc/1,cycle5/6).
