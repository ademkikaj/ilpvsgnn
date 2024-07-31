output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(imdb(+A,+B,+C,-D)).
warmode(person(+id,+-node_id,+-name)).
warmode(movie(+id,+-node_id,+-movie)).
warmode(director(+id,+-node_id)).
warmode(actor(+id,+-node_id,+-name)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
