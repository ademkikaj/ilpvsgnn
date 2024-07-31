output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(university(+B,-C)).
warmode(students(+id,+node_id,+-ranking)).
warmode(course(+id,+-course_id,+-diff,+-rating)).
warmode(professor(+id,+-prof_id,+-teaching,+-pop)).
warmode(registered(+id,+-course_id,+-grade,+-satis)).
warmode(ra(+id,+-prof_id,+-sal,+-cap)).
warmode(edge(+id,+-node_id,+-node_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
