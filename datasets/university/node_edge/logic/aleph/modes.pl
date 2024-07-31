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
:- modeb(*,course(+puzzle, +course_id, -diff, -rating)).
:- modeb(*,professor(+puzzle, -prof_id, -teaching, -pop)).
:- modeb(*,professor(+puzzle, +prof_id, -teaching, -pop)).
:- modeb(*,registered(+puzzle,+node_id,+course_id, -grade, -satis)).
:- modeb(*,registered(+puzzle,-node_id,+course_id, -grade, -satis)).
:- modeb(*,registered(+puzzle,+node_id,-course_id, -grade, -satis)).
:- modeb(*,registered(+puzzle,-node_id,-course_id, -grade, -satis)).
:- modeb(*,ra(+puzzle,+node_id,+prof_id, -sal, -cap)).
:- modeb(*,ra(+puzzle,-node_id,+prof_id, -sal, -cap)).
:- modeb(*,ra(+puzzle,+node_id,-prof_id, -sal, -cap)).
:- modeb(*,ra(+puzzle,-node_id,-prof_id, -sal, -cap)).
:- determination(university/1,students/3).
:- determination(university/1,course/4).
:- determination(university/1,professor/4).
:- determination(university/1,registered/5).
:- determination(university/1,ra/5).
