max_vars(10).
max_body(8).
head_pred(mutag,1).
body_pred(drug,5).
body_pred(c,2).
body_pred(n,2).
body_pred(o,2).
body_pred(h,2).
body_pred(cl,2).
body_pred(f,2).
body_pred(br,2).
body_pred(i,2).
body_pred(first,3).
body_pred(second,3).
body_pred(third,3).
body_pred(fourth,3).
body_pred(fifth,3).
body_pred(seventh,3).
type(mutag,(id,)).
type(drug,(id,node_id,ind1,inda,logp,lumo)).
type(c,(id,node_id)).
type(n,(id,node_id)).
type(o,(id,node_id)).
type(h,(id,node_id)).
type(cl,(id,node_id)).
type(f,(id,node_id)).
type(br,(id,node_id)).
type(i,(id,node_id)).
type(first(id,node_id,node_id)).
type(second(id,node_id,node_id)).
type(third(id,node_id,node_id)).
type(fourth(id,node_id,node_id)).
type(fifth(id,node_id,node_id)).
type(seventh(id,node_id,node_id)).
