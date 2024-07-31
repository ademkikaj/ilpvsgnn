:- aleph_set(clauselength,10).
:- aleph_set(c,3).
:- aleph_set(i,6).
:- aleph_set(verbosity,0).
:- aleph_set(minacc,0.05).
:- aleph_set(minpos,2).
:- aleph_set(nodes,30000).
:- modeh(1,mutag(+puzzle)).
:- modeb(*,atom(+puzzle, -node_id, -charge)).
:- modeb(*,drug(+puzzle, -node_id, -ind1, -inda, -logp, -lumo)).
:- modeb(*,bond(+puzzle, -node_id,-bond_type)).
:- modeb(*,nitro(+puzzle, -node_id)).
:- modeb(*,benzene(+puzzle, -node_id)).
:- modeb(*,edge(+puzzle, -object, -object)).
:- modeb(*,edge(+puzzle, -object, +object)).
:- modeb(*,edge(+puzzle, +object, -object)).
:- determination(mutag/1,atom/3).
:- determination(mutag/1,drug/6).
:- determination(mutag/1,bond/3).
:- determination(mutag/1,nitro/2).
:- determination(mutag/1,benzene/2).
:- determination(mutag/1,edge/3).
