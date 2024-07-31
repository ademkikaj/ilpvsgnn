output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(imdb(+A,+B,+C,-D)).
warmode(person(+id,+-node_id,+-name)).
warmode(movie(+id,+-node_id,+-movie)).
warmode(genre(+id,+-node_id,+-genre)).
warmode(gender(+id,+-node,+-gender)).
warmode(director(+id,+-node_id)).
warmode(actor(+id,+-node_id)).
warmode(edge(+id,+-node_id,+-node_id)).
typed_language(yes).
type(imdb,(id,name,name,class)).
type(movie,(id,node_id,movie)).
type(genre,(id,node_id,genre)).
type(gender(id,node_id,gender)).
type(actor,(id,node_id)).
type(edge,(id,node_id,node_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
