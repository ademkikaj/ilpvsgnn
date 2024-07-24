:- style_check(-singleton).
cyclic(A,pos) :- node(A,B),green(B), !.
cyclic(A,neg).
