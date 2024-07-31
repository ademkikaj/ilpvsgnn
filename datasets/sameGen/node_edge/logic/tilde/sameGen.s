output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(sameGen(+A,+B,+C,-D)).
warmode(person(+puzzle,-name)).
warmode(parent(+puzzle,-name,-name)).
warmode(same_gen(+puzzle,-name,-name)).
auto_lookahead(person(Id,Name),[Name]).
auto_lookahead(parent(Id,Parent,Name),[Parent,Name]).
auto_lookahead(same_gen(Id,Name1,Name2),[Name1,Name2]).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
