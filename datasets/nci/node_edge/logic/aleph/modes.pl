:- use_module(library(aleph)).
:- if(current_predicate(use_rendering/1)).
:- use_rendering(prolog).
:- endif.
:- aleph.
:- style_check(-discontiguous).
:- aleph_set(verbosity,1).
:- aleph_set(minpos,5).
:- aleph_set(nodes,100000).
:- modeh(1,nci(+id)).
:- aleph_set(clauselength,10).
:- modeb(*,atom(+id, -node_id, -atom_type)).
:- modeb(*,bond(+id, -node_id, -node_id, -bond_type)).
:- modeb(*,nitro(+id, -node_id, -node_id)).
:- modeb(*,benzene(+id, +node_id, +node_id)).
:- modeb(*,eq_atom(+atom_type, -atom_type)).
:- modeb(*,eq_bond(+bond_type, -bond_type)).
:- determination(nci/1,atom/3).
:- determination(nci/1,bond/4).
:- determination(nci/1,eq_atom/2).
:- determination(nci/1,eq_bond/2).
