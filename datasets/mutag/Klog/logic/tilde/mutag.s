output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(mutag(+B,-C)).
warmode(atom(+id,+node_id,+-atom_type,+-charge)).
warmode(bond(+id,+-node_id)).
warmode(nitro(+id,+-node_id)).
warmode(benzene(+id,+-node_id)).
warmode(edge(+id,+-node_id,+-node_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
