output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(university(+B,-C)).
warmode(students(+id,+-node_id,+-ranking)).
warmode(course(+id,+-course_id,+-diff,+-rating)).
warmode(professor(+id,+-prof_id,+-teaching,+-pop)).
warmode(registered(+id,+-node_id,+-course_id,+-grade,+-satis)).
warmode(ra(+id,+-node_id,+-prof_id,+-sal,+-cap)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
