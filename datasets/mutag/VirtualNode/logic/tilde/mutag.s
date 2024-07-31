output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(mutag(+B,-C)).
warmode(drug(+id,+-ind1,+-inda,+-logp,+-lumo)).
warmode(c(+id,+-node_id)).
warmode(n(+id,+-node_id)).
warmode(o(+id,+-node_id)).
warmode(h(+id,+-node_id)).
warmode(cl(+id,+-node_id)).
warmode(f(+id,+-node_id)).
warmode(br(+id,+-node_id)).
warmode(i(+id,+-node_id)).
warmode(first(+id,+-node_id,+-node_id)).
warmode(second(+id,+-node_id,+-node_id)).
warmode(third(+id,+-node_id,+-node_id)).
warmode(fourth(+id,+-node_id,+-node_id)).
warmode(fifth(+id,+-node_id,+-node_id)).
warmode(seventh(+id,+-node_id,+-node_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
