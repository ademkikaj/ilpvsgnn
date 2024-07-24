output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
max_lookahead(1).
predict(cyclic(+B,-C)).
warmode(edge(+id_int,-id_int)).
warmode(node(+id,+-color)).
warmode(green(+color)).
warmode(red(+color)).
auto_lookahead(edge(Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).
auto_lookahead(node(Node_id,Color),[Node_id,Color]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
