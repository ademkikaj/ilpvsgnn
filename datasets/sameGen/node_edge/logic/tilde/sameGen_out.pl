:- style_check(-singleton).
sameGen(A,B,C,pos) :- same_gen(A,D,E), !.
sameGen(A,B,C,neg).
