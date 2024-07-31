output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
exhaustive_lookahead(1).
predict(krk(+B,-C)).
warmode(white_king(+id,-node_id,#x,#y)).
warmode(white_rook(+id,-node_id,#x,#y)).
warmode(black_king(+id,-node_id,#x,#y)).
warmode(edge(+id,-node_id,-node_id)).
warmode(same_rank(+id,+-node_id,+-node_id)).
warmode(same_file(+id,+-node_id,+-node_id)).
warmode(distance(white_king(+id,+node_id,+x,+y),black_king(+id,+node_id,+x,+y),#d)).
warmode(one(#d)).
auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id,X,Y]).
auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id,X,Y]).
auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id,X,Y]).
auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).
auto_lookahead(same_rank(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).
auto_lookahead(same_file(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).
auto_lookahead(distance(white_king(Id,Node_id,X,Y),black_king(Id,Node_id,X,Y),D),[D]).
auto_lookahead(one(D),[D]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
