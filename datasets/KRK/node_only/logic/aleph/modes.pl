:- use_module(library(aleph)).
:- if(current_predicate(use_rendering/1)).
:- use_rendering(prolog).
:- endif.
:- aleph.
:- style_check(-discontiguous).
:- aleph_set(verbosity,1).
:- aleph_set(minpos,5).
:- aleph_set(nodes,100000).
:- modeh(1,krk(+id)).
:- modeb(*,white_king(+id, -node_id, #x, #y)).
:- modeb(*,white_rook(+id, -node_id,#x, #y)).
:- modeb(*,black_king(+id, -node_id, #x, #y)).
:- modeb(*,edge(+id, -node_id, -node_id)).
:- determination(krk/1,white_king/4).
:- determination(krk/1,white_rook/4).
:- determination(krk/1,black_king/4).
:- determination(krk/1,edge/3).
