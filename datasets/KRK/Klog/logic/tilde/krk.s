output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(krk(+B,-C)).
warmode(white_king(+id,+node_id,#x,#y)).
warmode(white_rook(+id,+node_id,#x,#y)).
warmode(black_king(+id,+node_id,#x,#y)).
warmode(edge(+id,-node_id,-node_id)).
typed_language(yes).
auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id,X,Y]).
auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id,X,Y]).
auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id,X,Y]).
auto_lookahead(edge(Id,Node_id1),[Node_id1]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
