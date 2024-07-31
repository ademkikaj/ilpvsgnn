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
:- modeb(*,white_king(+id, -object, #x, #y)).
:- modeb(*,white_rook(+id, -object,#x, #y)).
:- modeb(*,black_king(+id, -object, #x, #y)).
:- modeb(*,edge(+id, -object, -object)).
:- modeb(*,same_rank(+id, -object, -object)).
:- modeb(*,same_file(+id, -object, -object)).
:- modeb(*,instance(+id, white_king, -object)).
:- modeb(*,instance(+id, white_rook, -object)).
:- modeb(*,instance(+id, black_king, -object)).
:- modeb(*,same_x(+x, #x)).
:- modeb(*,same_y(+y, #y)).
:- determination(krk/1,white_king/4).
:- determination(krk/1,white_rook/4).
:- determination(krk/1,black_king/4).
:- determination(krk/1,edge/3).
:- determination(krk/1,same_rank/3).
:- determination(krk/1,same_file/3).
:- determination(krk/1,instance/3).
:- determination(krk/1,same_x/2).
:- determination(krk/1,same_y/2).
