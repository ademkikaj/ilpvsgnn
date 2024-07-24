output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
exhaustive_lookahead(1).
query_batch_size(50000).
max_lookahead(1).
predict(nci(+id,-B)).
rmode(atom(+id,+node_id,-atom_type)).
rmode(bond(+id,+-node_id_1,+-node_id_2,-bond_type)).
rmode(eq_atom(+X,#[o,n,c,s,cl,p,f,na,sn,pt,ni,zn,mn,br,y,nd,ti,eu,cu,tl,zr,hf,in,ga,k,si,i])).
rmode(eq_bond(+X,#[1,2,3])).
rmode(connected(+id,+node_id_1,+node_id_2)).
typed_language(yes).
type(nci(id,class)).
type(atom(id,node_id,atom_type)).
type(bond(id,node_id,node_id,bond_type)).
type(eq_atom(atom_type,atom_type)).
type(eq_bond(bond_type,bond_type)).
type(connected(id,node_id,node_id)).
auto_lookahead(atom(Id,Node,Atom),[Node,Atom]).
auto_lookahead(bond(Id,Node_id_1,Node_id_2,Bond_type),[Node_id_1,Node_id_2,Bond_type]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
