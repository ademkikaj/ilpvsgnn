:- style_check(-singleton).
mutag(A,pos) :-   nitro(A,B,C), nitro(A,D,E), benzene(A,E,F), benzene(A,F,C), !.
mutag(A,pos) :-   nitro(A,B,C), benzene(A,B,C), !.
mutag(A,pos) :-   nitro(A,B,C), nitro(A,D,E), benzene(A,C,F), benzene(A,G,D), !.
mutag(A,neg).
