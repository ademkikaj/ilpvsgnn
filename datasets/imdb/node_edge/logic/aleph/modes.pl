:- aleph_set(i,3).
:- aleph_set(verbosity,0).
:- aleph_set(clauselength,6).
:- aleph_set(minacc,0.1).
:- aleph_set(minpos,3).
:- aleph_set(nodes,20000).
:- aleph_set(noise,5).
:- aleph_set(c,3).
:- modeh(1,imdb(+puzzle)).
:- modeb(*,students(+puzzle, -node_id, -ranking)).
:- modeb(*,course(+puzzle, -course_id, -diff, -rating)).
:- modeb(*,professor(+puzzle, -prof_id, -teaching, -pop)).
:- modeb(*,registered(+puzzle, -course_id, -grade, -satis)).
:- modeb(*,ra(+puzzle, -prof_id, -sal, -cap)).
:- modeb(*,edge(+puzzle, -object, -object)).
:- modeb(*,edge(+puzzle, -object, +object)).
:- modeb(*,edge(+puzzle, +object, -object)).
:- determination(imdb/1,students/3).
:- determination(imdb/1,course/4).
:- determination(imdb/1,professor/4).
:- determination(imdb/1,registered/4).
:- determination(imdb/1,RA/4).
:- determination(imdb/1,edge/3).
