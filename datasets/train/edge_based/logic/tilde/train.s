output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(train(+B,-C)).
warmode(has_car(+id,+-node_id,+-node_id)).
warmode(has_load(+id,+-node_id,+-node_id)).
warmode(edge(+id,+-node_id,+-object)).
warmode(instance(+id,+-node_id)).
warmode(train(+object)).
warmode(car(+object)).
warmode(load(+object)).
warmode(short(+object)).
warmode(long(+object)).
warmode(two_wheels(+object)).
warmode(three_wheels(+object)).
warmode(roof_open(+object)).
warmode(roof_closed(+object)).
warmode(zero_load(+object)).
warmode(one_load(+object)).
warmode(two_load(+object)).
warmode(three_load(+object)).
warmode(circle(+object)).
warmode(triangle(+object)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
