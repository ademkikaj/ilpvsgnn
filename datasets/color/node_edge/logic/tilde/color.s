output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
max_lookahead(4).
exhaustive_lookahead(2).
predict(color(+id,+node_id,-C)).
warmode(node(+id,+node_id,-color)).
warmode(edge(+id,-node_id,-node_id)).
auto_lookahead(node(Id,Node_id,Color),[Node_id,Color]).
auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
