output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(krk(+B,-C)).
rmode(white_king(+id,-node_id,-x,-y)).
rmode(white_rook(+id,-node_id,-x,-y)).
rmode(black_king(+id,-node_id,-x,-y)).
rmode(edge(+id,-node_id,-node_id)).
typed_language(yes).
type(krk(id,class)).
type(white_king(id,node_id,x,y)).
type(white_rook(id,node_id,x,y)).
type(black_king(id,node_id,x,y)).
type(edge(id,node_id,node_id)).
auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id]).
auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id]).
auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id]).
auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
