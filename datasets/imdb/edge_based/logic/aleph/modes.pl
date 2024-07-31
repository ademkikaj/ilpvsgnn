:- aleph_set(i,3).
:- aleph_set(verbosity,0).
:- aleph_set(clauselength,6).
:- aleph_set(minacc,0.1).
:- aleph_set(minpos,3).
:- aleph_set(nodes,20000).
:- aleph_set(noise,5).
:- aleph_set(c,3).
:- modeh(1,imdb(+puzzle)).
:- modeb(*,white_king(+puzzle, -node_id, -x, -y)).
:- modeb(*,white_rook(+puzzle, -node_id, -x, -y)).
:- modeb(*,black_king(+puzzle, -node_id, -x, -y)).
:- modeb(*,edge(+puzzle, -object, -object)).
:- modeb(*,edge(+puzzle, -object, +object)).
:- modeb(*,edge(+puzzle, +object, -object)).
:- modeb(*,same_rank(+puzzle, -object, -object)).
:- modeb(*,same_rank(+puzzle, -object, +object)).
:- modeb(*,same_rank(+puzzle, +object, -object)).
:- modeb(*,same_file(+puzzle, -object, -object)).
:- modeb(*,same_file(+puzzle, -object, +object)).
:- modeb(*,same_file(+puzzle, +object, -object)).
:- modeb(*,instance(+puzzle, white_king, -object)).
:- modeb(*,instance(+puzzle, white_rook, -object)).
:- modeb(*,instance(+puzzle, black_king, -object)).
:- determination(imdb/1,white_king/4).
:- determination(imdb/1,white_rook/4).
:- determination(imdb/1,black_king/4).
:- determination(imdb/1,edge/3).
:- determination(imdb/1,same_rank/3).
:- determination(imdb/1,same_file/3).
:- determination(imdb/1,instance/3).
