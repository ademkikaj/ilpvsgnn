:- use_module(library(aleph)).
:- if(current_predicate(use_rendering/1)).
:- use_rendering(prolog).
:- endif.
:- aleph.
:- style_check(-discontiguous).
:- aleph_set(verbosity,1).
:- aleph_set(minpos,5).
:- aleph_set(nodes,100000).
:- modeh(1,mutag(+id)).
:- aleph_set(clauselength,15).
:- modeb(*,atom(+id, -object,-atom_type, -charge)).
:- modeb(*,bond(+id, -object, -object,-bond_type)).
:- modeb(*,nitro(+id, -object, -object)).
:- modeb(*,benzene(+id, -object, -object)).
:- modeb(*,eq_atom(+atom_type, -atom_type)).
:- modeb(*,eq_bond(+bond_type, -bond_type)).
:- modeb(*,eq_charge(+charge, -charge)).
:- modeb(*,lteq(+charge, -charge)).
:- modeb(*,gteq(+charge, -charge)).
:- determination(mutag/1,atom/4).
:- determination(mutag/1,bond/4).
:- determination(mutag/1,nitro/3).
:- determination(mutag/1,benzene/3).
:- determination(mutag/1,eq_atom/2).
:- determination(mutag/1,eq_bond/2).
:- determination(mutag/1,eq_charge/2).
