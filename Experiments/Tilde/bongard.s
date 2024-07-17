
output_options([c45,c45c,c45e,lp,prolog]).
%talking(4).
use_packs(ilp).
transform_queries(once).
write_predictions([testing]).

warmr_maxdepth(3).

predict(bongard(+B,-C)).

warmr_assoc([warmr_rules,warmr_rules_min_confidence(0.5),warmr_rules_min_support(0.1)]).
warmr_assoc_output_options([assoc_pred]).

rmode(5: triangle(+P,+-S)).
rmode(5: square(+P,+-S)).
rmode(5: circle(+P,+-S)).
rmode(5: in(+P,+S1,+-S2)).


typed_language(yes).
type(triangle(pic,obj)).
type(square(pic,obj)).
type(circle(pic,obj)).
type(in(pic,obj,obj)).
type(bongard(pic,class)).

max_lookahead(2).
lookahead(triangle(X,Y), in(X,Y,-Z)).
%lookahead(triangle(X,Y), in(X,-Z,Y)).
lookahead(square(X,Y), in(X,Y,-Z)).
%lookahead(square(X,Y), in(X,-Z,Y)).
lookahead(circle(X,Y), in(X,Y,-Z)).
%lookahead(circle(X,Y), in(X,-Z,Y)).

lookahead(in(X,Y,Z), triangle(X,Z)).
lookahead(in(X,Y,Z), square(X,Z)).
lookahead(in(X,Y,Z), circle(X,Z)).


write_predictions([testing, distribution]).

combination_rule([product, sum]).

%random_test_set(0.2).
execute(t).

%execute(nfold(tilde,3)).
execute(q).


