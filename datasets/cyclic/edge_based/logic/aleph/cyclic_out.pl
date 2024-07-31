:- style_check(-singleton).
cyclic(A,pos):-edge(A,B),instance(B),instance(A), !.
cyclic(C,pos):-edge(C,C), !.

cyclic(A,neg).
