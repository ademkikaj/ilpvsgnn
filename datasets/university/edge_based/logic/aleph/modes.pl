:- aleph_set(clauselength,10).
:- aleph_set(c,3).
:- aleph_set(i,6).
:- aleph_set(verbosity,0).
:- aleph_set(minacc,0.05).
:- aleph_set(minpos,2).
:- aleph_set(nodes,30000).
:- modeh(1,university(+puzzle)).
:- modeb(*,students(+puzzle, -node_id, -ranking)).
:- modeb(*,course(+puzzle, -course_id, -diff)).
:- modeb(*,course(+puzzle, +course_id, -diff)).
:- modeb(*,professor(+puzzle, -prof_id, -teaching)).
:- modeb(*,professor(+puzzle, +prof_id, -teaching)).
:- modeb(*,registered(+puzzle,+node_id,+course_id, -grade)).
:- modeb(*,registered(+puzzle,-node_id,+course_id, -grade)).
:- modeb(*,registered(+puzzle,+node_id,-course_id, -grade)).
:- modeb(*,registered(+puzzle,-node_id,-course_id, -grade)).
:- modeb(*,ra(+puzzle,+node_id,+prof_id, -sal)).
:- modeb(*,ra(+puzzle,-node_id,+prof_id, -sal)).
:- modeb(*,ra(+puzzle,+node_id,-prof_id, -sal)).
:- modeb(*,ra(+puzzle,-node_id,-prof_id, -sal)).
:- determination(university/1,students/3).
:- determination(university/1,course/3).
:- determination(university/1,professor/3).
:- determination(university/1,registered/4).
:- determination(university/1,ra/4).
