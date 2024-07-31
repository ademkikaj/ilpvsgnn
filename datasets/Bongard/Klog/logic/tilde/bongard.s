output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(bongard(+id,-C)).
rmode(triangle(+id,-object)).
rmode(square(+id,-object)).
rmode(circle(+id,-object)).
rmode(edge(+id,-object,-object)).
rmode(in(+id,-object)).
typed_language(yes).
type(bongard(pic,class)).
type(triangle(pic,obj)).
type(square(pic,obj)).
type(circle(pic,obj)).
type(edge(pic,obj,obj)).
type(in(pic,obj)).
auto_lookahead(triangle(Id,Obj),[Obj]).
auto_lookahead(square(Id,Obj),[Obj]).
auto_lookahead(circle(Id,Obj),[Obj]).
auto_lookahead(edge(Id,Obj1,Obj2),[Obj1,Obj2]).
auto_lookahead(in(Id,Obj),[Obj]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
