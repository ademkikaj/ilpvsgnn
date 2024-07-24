:- style_check(-singleton).
sameGen(A,B,C,pos) :- same_gen(A,D), !.
sameGen(A,B,C,neg).
