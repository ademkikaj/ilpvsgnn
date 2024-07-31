output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
exhaustive_lookahead(1).
max_lookahead(3).
query_batch_size(50000).
predict(cancer(+B,-C)).
typed_language(yes).
type(atom(id,node_id,atom_type,charge)).
type(bond(id,node_id,node_id,bond_type)).
type(eq_atomtype(atom_type,atom_type)).
type(eq_charge(charge,charge)).
type(eq_bondtype(bond_type,bond_type)).
rmode(atom(+drug_id,+node_id,-atom_type,-charge)).
rmode(bond(+drug_id,+node_id_1,+node_id_2,-bond_type)).
auto_lookahead(atom(Id,Node,Atom,Charge),[Node,Atom,Charge]).
auto_lookahead(bond(Id,Node_id_1,Node_id_2,Bond_type),[Node_id_1,Node_id_2,Bond_type]).
auto_lookahead(eq_atomtype(Atom_type,Atom_type),[Atom_type]).
auto_lookahead(eq_charge(Charge,Charge),[Charge]).
auto_lookahead(eq_bondtype(Bond_type,Bond_type),[Bond_type]).
rmode(eq_atomtype(+X, #[1,10,101,102,113,115,120,121,129,134,14,15,16,17,19,191,192,193,2,21,22,232,26,27,29,3,31,32,33,34,35,36,37,38,40,41,42,45,49,499,50,51,52,53,60,61,62,70,72,74,75,76,77,78,79,8,81,83,84,85,87,92,93,94,95,96])).
rmode(eq_charge(+X, #['a0=_0_0175<x<=0_0615','a0=_0_1355<x<=_0_0175','a0=_inf<x<=_0_1355','a0=0_0615<x<=0_1375','a0=0_1375<x<=+inf'])).
rmode(eq_bondtype(+X, #[1,2,3,7])).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
