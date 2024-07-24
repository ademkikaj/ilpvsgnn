output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(bongard(+B,-C)).
rmode(triangle(+P,+-S)).
rmode(square(+P,+-S)).
rmode(circle(+P,+-S)).
rmode(edge(+P,+S1,+-S2)).
typed_language(yes).
type(bongard(pic,class)).
type(triangle(pic,obj)).
type(square(pic,obj)).
type(circle(pic,obj)).
type(edge(pic,obj,obj)).
auto_lookahead(triangle(Id,S),[S]).
auto_lookahead(square(Id,S),[S]).
auto_lookahead(circle(Id,S),[S]).
auto_lookahead(edge(Id,S1,S2),[S1,S2]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
