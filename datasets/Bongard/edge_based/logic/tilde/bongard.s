output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(bongard(+B,-C)).
rmode(shape(+P,triangle,+-C)).
rmode(shape(+P,square,+-C)).
rmode(shape(+P,circle,+-C)).
rmode(in(+P,+S1,+-S2)).
rmode(instance(+P,+-C)).
typed_language(yes).
type(bongard(pic,class)).
type(in(pic,obj,obj)).
type(shape(pic,fact,obj)).
type(instance(pic,obj)).
auto_lookahead(shape(Id,T,C),[T,C]).
auto_lookahead(in(Id,S1,S2),[S1,S2]).
auto_lookahead(instance(Id,C),[C]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
