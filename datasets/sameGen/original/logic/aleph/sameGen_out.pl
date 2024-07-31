:- style_check(-singleton).
sameGen(A,B,pos) :-   same_gen(B,A), !.
sameGen(A,B,neg).
