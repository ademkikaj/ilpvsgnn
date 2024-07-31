output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(university(+A,-C,+B)).
warmode(course(+-course_id,+-diff,+-rating)).
warmode(prof(+-prof_id,+-teaching,+-pop)).
warmode(registration(+-student_id,+-course_id,+-grade,+-satis)).
warmode(ra(+-student_id,+-prof_id,+-sal,+-cap)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
