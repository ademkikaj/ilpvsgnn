output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(sameGen(+A,+B,-C)).
warmode(same_gen(+A,-B)).
warmode(person(+A)).
warmode(parent(+A,-B)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
