output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(mutag(+B,-C)).
warmode(instance(+id,+atom,+-atom_type)).
warmode(bond(+id,+-node_id,+-node_id,+-bond_type)).
warmode(nitro(+id,+-node_id,+-node_id)).
warmode(benzene(+id,+-node_id,+-node_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
