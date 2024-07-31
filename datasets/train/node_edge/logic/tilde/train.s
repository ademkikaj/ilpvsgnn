output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(train(+B,-C)).
warmode(has_car(+id,+-car_id)).
warmode(has_load(+id,+-car_id,+-load_id)).
warmode(short(+id,+-car_id)).
warmode(long(+id,+-car_id)).
warmode(two_wheels(+id,+-car_id)).
warmode(three_wheels(+id,+-car_id)).
warmode(roof_open(+id,+-car_id)).
warmode(roof_closed(+id,+-car_id)).
warmode(zero_load(+id,+-load_id)).
warmode(one_load(+id,+-load_id)).
warmode(two_load(+id,+-load_id)).
warmode(three_load(+id,+-load_id)).
warmode(circle(+id,+-load_id)).
warmode(triangle(+id,+-load_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
