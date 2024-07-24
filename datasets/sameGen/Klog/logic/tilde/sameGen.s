output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(sameGen(+A,+B,+C,-D)).
warmode(person(+id,+-node_id,+-name)).
warmode(parent(+id,+-node_id)).
warmode(same_gen(+id,+-node_id)).
warmode(edge(+id,+-node_id,+-node_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
