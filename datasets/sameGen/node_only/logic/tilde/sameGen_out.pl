:- style_check(-singleton).
sameGen(A,B,C,neg) :- parent(A,D,E), !.
sameGen(A,B,C,pos).
