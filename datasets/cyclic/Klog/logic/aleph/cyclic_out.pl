:- style_check(-singleton).
cyclic(A,pos):-edge(A,B),edge(B,A), !.

cyclic(A,neg).
