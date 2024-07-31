output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
max_lookahead(1).
exhaustive_lookahead(1).
query_batch_size(50000).
predict(bongard(+B,-C)).
rmode(triangle(+id,-object)).
rmode(square(+id,-object)).
rmode(circle(+id,-object)).
rmode(in(+id,-object)).
rmode(edge(+id,-object,-object)).
typed_language(yes).
type(bongard(pic,class)).
type(triangle(pic,obj)).
type(square(pic,obj)).
type(circle(pic,obj)).
type(in(pic,obj)).
type(edge(pic,obj,obj)).
auto_lookahead(triangle(Id,Obj),[Obj]).
auto_lookahead(square(Id,Obj),[Obj]).
auto_lookahead(circle(Id,Obj),[Obj]).
auto_lookahead(in(Id,Obj),[Obj]).
auto_lookahead(edge(Id,Obj1,Obj2),[Obj1,Obj2]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
