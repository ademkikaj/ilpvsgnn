max_vars(8).
max_body(8).


head_pred(f,1).
body_pred(distance,3).
body_pred(square,4).
body_pred(king,1).
body_pred(rook,1).
body_pred(white,1).
body_pred(black,1).
body_pred(one,1).

type(f,(state,)).
type(square,(state, pos, color, piecetype)).
type(distance,(pos, pos, integer)).
type(king,(piecetype,)).
type(rook,(piecetype,)).
type(white,(color,)).
type(black,(color,)).
type(one,(integer,)).

direction(f,(in,)).
direction(distance,(in, in, out)).
direction(square,(in, out, out, out)).
direction(king,(in,)).
direction(rook,(in,)).
direction(white,(in,)).
direction(black,(in,)).
direction(one,(in,)).