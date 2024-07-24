output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
max_lookahead(1).
query_batch_size(50000).
predict(mutag(+B,-C)).
warmode(atom(+id,-node_id,-atom_type,-charge)).
warmode(bond(+id,-node_id,-node_id,-bond_type)).
warmode(nitro(+id,-node_id,-node_id)).
warmode(benzene(+id,+-node_id,+-node_id)).
warmode(eq_atom(+atom_type,-atom_type)).
warmode(eq_bond(+bond_type,-bond_type)).
warmode(eq_charge(+charge,-charge)).
warmode(lteq(+charge,-charge)).
warmode(gteq(+charge,-charge)).
auto_lookahead(atom(Id,Node,Atom,Charge),[Node,Atom,Charge]).
auto_lookahead(bond(Id,Node_id_1,Node_id_2,Bond_type),[Node_id_1,Node_id_2,Bond_type]).
auto_lookahead(nitro(Id,Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).
auto_lookahead(benzene(Id,Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
