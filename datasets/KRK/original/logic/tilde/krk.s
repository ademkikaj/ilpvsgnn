output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
write_predictions([testing]).

predict(krk(+id,-class)).
rmode(5: king(+-piece)).
rmode(5: rook(+-piece)).
rmode(5: white(+-color)).
rmode(5: black(+-color)).
rmode(5: cell(+id,(+-x,+-y),+-color,+-piecetype)).
rmode(5: one(-integer)).

typed_language(yes).
type(krk(id,class)).
type(king(piece)).
type(rook(piece)).
type(white(color)).
type(black(color)).
type(distance((int,int),(int,int),integer)).
type(cell(id,(int,int),color,piecetype)).
type(one(int)).

combination_rule([product, sum]).

random_test_set(0.5).
execute(t).
execute(q).

