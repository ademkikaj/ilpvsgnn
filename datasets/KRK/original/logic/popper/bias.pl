max_vars(8).
max_body(8).
max_clauses(1).
head_pred(krk,1).
body_pred(distance,3).
body_pred(cell,4).
body_pred(king,1).
body_pred(rook,1).
body_pred(white,1).
body_pred(black,1).
body_pred(one,1).
type(krk,(state,)).
type(distance,(pos, pos, integer)).
type(cell,(state, pos, color, piecetype)).
type(king,(piecetype,)).
type(rook,(piecetype,)).
type(white,(color,)).
type(black,(color,)).
type(one,(integer,)).
direction(krk,(in,)).
direction(distance,(in, in, out)).
direction(cell,(in, out, out, out)).
direction(king,(in,)).
direction(rook,(in,)).
direction(white,(in,)).
direction(black,(in,)).
direction(one,(in,)).
