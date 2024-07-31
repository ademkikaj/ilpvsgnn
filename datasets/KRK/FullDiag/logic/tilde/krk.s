output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(krk(+B,-C)).
warmode(white_king(+id,+-node_id)).
warmode(white_rook(+id,+-node_id)).
warmode(black_king(+id,+-node_id)).
warmode(edge(+id,+-node_id,+-node_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
