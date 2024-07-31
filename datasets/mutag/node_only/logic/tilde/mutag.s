output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(mutag(+B,-C)).
warmode(atom(+id,+-node_id,+-charge)).
warmode(drug(+id,+-node_id,+-ind1,+-inda,+-logp,+-lumo)).
warmode(bond(+id,+-node_id,+-bond_type)).
warmode(nitro(+id,+-node_id)).
warmode(benzene(+id,+-node_id)).
warmode(edge(+id,+-node_id,+-node_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
