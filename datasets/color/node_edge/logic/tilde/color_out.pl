:- style_check(-singleton).
color(A,B,pos) :- edge(A,C,D), !.
color(A,B,neg).
