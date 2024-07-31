:- style_check(-singleton).
cyclic(A,pos):-edge(A,B),edge(B,A), !.
cyclic(C,pos):-edge(C,C), !.

cyclic(A,neg).
