:- aleph_set(clauselength,10).
:- aleph_set(c,3).
:- aleph_set(i,6).
:- aleph_set(verbosity,0).
:- aleph_set(minacc,0.05).
:- aleph_set(minpos,2).
:- aleph_set(nodes,30000).
:- modeh(1,mutag(+puzzle)).
:- modeb(*,instance(+puzzle,atom,-object)).
:- modeb(*,bond(+puzzle, -object, -object,+bond_type)).
:- modeb(*,bond(+puzzle, -object, +object,+bond_type)).
:- modeb(*,bond(+puzzle, +object, -object,+bond_type)).
:- modeb(*,nitro(+puzzle, -object, -object)).
:- modeb(*,nitro(+puzzle, -object, +object)).
:- modeb(*,nitro(+puzzle, +object, -object)).
:- modeb(*,benzene(+puzzle, -object, -object)).
:- modeb(*,benzene(+puzzle, -object, +object)).
:- modeb(*,benzene(+puzzle, +object, -object)).
:- determination(mutag/1,atom/3).
:- determination(mutag/1,bond/4).
:- determination(mutag/1,nitro/3).
:- determination(mutag/1,benzene/3).
