:- use_module(library(aleph)).
:- if(current_predicate(use_rendering/1)).
:- use_rendering(prolog).
:- endif.
:- aleph.
:- style_check(-discontiguous).
:- aleph_set(verbosity,1).
:- aleph_set(minpos,5).
:- aleph_set(nodes,100000).
:- modeh(1,cancer(+id)).
:- aleph_set(clauselength,15).
:- modeb(*,atom(+id, -node_id, -atom_type, -charge)).
:- modeb(*,bond(+id, -node_id, -node_id, -bond_type)).
:- modeb(*,eq_atomtype(+atom_type, -atom_type)).
:- modeb(*,eq_charge(+charge, -charge)).
:- modeb(*,eq_bondtype(+bond_type, -bond_type)).
:- modeb(*,cycle3(+id, -node_id, -node_id, -node_id)).
:- modeb(*,cycle4(+id, -node_id, -node_id, -node_id, -node_id)).
:- determination(cancer/1,atom/4).
:- determination(cancer/1,bond/4).
:- determination(cancer/1,eq_atomtype/2).
:- determination(cancer/1,eq_charge/2).
:- determination(cancer/1,eq_bondtype/2).
:- determination(cancer/1,cycle3/4).
:- determination(cancer/1,cycle4/5).
