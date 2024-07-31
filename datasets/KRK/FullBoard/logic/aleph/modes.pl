:- aleph_set(clauselength,10).
:- aleph_set(c,3).
:- aleph_set(i,6).
:- aleph_set(verbosity,0).
:- aleph_set(minacc,0.05).
:- aleph_set(minpos,2).
:- aleph_set(nodes,30000).
:- modeh(1,krk(+puzzle)).
:- modeb(*,white_king(+puzzle, -node_id)).
:- modeb(*,white_rook(+puzzle, -node_id)).
:- modeb(*,black_king(+puzzle, -node_id)).
:- modeb(*,edge(+puzzle, -object, -object)).
:- modeb(*,edge(+puzzle, -object, +object)).
:- modeb(*,edge(+puzzle, +object, -object)).
:- determination(krk/1,white_king/2).
:- determination(krk/1,white_rook/2).
:- determination(krk/1,black_king/2).
:- determination(krk/1,edge/3).
