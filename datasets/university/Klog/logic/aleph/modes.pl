:- aleph_set(clauselength,10).
:- aleph_set(c,3).
:- aleph_set(i,6).
:- aleph_set(verbosity,0).
:- aleph_set(minacc,0.05).
:- aleph_set(minpos,2).
:- aleph_set(nodes,30000).
:- modeh(1,university(+puzzle)).
:- modeb(*,students(+puzzle, -node_id, -ranking)).
:- modeb(*,course(+puzzle, -course_id, -diff, -rating)).
:- modeb(*,professor(+puzzle, -prof_id, -teaching, -pop)).
:- modeb(*,registered(+puzzle, -course_id, -grade, -satis)).
:- modeb(*,ra(+puzzle, -prof_id, -sal, -cap)).
:- modeb(*,edge(+puzzle, -object, -object)).
:- modeb(*,edge(+puzzle, -object, +object)).
:- modeb(*,edge(+puzzle, +object, -object)).
:- determination(university/1,students/3).
:- determination(university/1,course/4).
:- determination(university/1,professor/4).
:- determination(university/1,registered/4).
:- determination(university/1,ra/4).
:- determination(university/1,edge/3).
