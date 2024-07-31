:- style_check(-singleton).
f(A) :-   edge(A,B), edge(B,C), edge(C,D), edge(D,A), !.
cyclic(A,neg).
