eq_atom(A,B) :- A == B.
eq_bond(A,B) :- A == B.
eq_charge(A,B) :- A == B.
lteq(X,Y) :- X =< Y, !.
gteq(X,Y) :- X >= Y, !.

