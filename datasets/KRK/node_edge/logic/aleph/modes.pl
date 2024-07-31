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
:- modeb(*,white_king(+id, -node_id, -x, -y)).
:- modeb(*,white_rook(+id, -node_id, -x, -y)).
:- modeb(*,black_king(+id, -node_id, -x, -y)).
:- modeb(*,edge(+id, -node_id, -node_id)).
:- modeb(*,same_rank(+id, -node_id, -node_id)).
:- modeb(*,same_file(+id, -node_id, -node_id)).
:- modeb(*,same_x(+x, #[0,1,2,3,4,5,6,7])).
:- modeb(*,same_y(+y, #[0,1,2,3,4,5,6,7])).
:- modeb(*,distance(+x,+y,+x,+y,-d)).
:- modeb(*,one(+d)).
:- determination(krk/1,white_king/4).
:- determination(krk/1,white_rook/4).
:- determination(krk/1,black_king/4).
:- determination(krk/1,edge/3).
:- determination(krk/1,same_rank/3).
:- determination(krk/1,same_file/3).
:- determination(krk/1,same_x/2).
:- determination(krk/1,same_y/2).
:- determination(krk/1,distance/5).
:- determination(krk/1,one/1).
