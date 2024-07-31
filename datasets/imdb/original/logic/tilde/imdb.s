output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(imdb(+A,+B,-C)).
warmode(movie(+movie,+-person)).
warmode(actor(+person)).
warmode(director(+person)).
warmode(gender(+person,+-gender)).
warmode(genre(+person,+-genre)).
typed_language(yes).
type(imdb,(person,person,class)).
type(movie,(movie,person)).
type(actor,(person,)).
type(director,(person,)).
type(genre,(person,genre)).
type(gender,(person,gender)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
