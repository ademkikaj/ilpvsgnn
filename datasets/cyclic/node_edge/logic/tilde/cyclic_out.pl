:- style_check(-singleton).
cyclic(A,pos) :- edge(A,B),edge(B,C),edge(C,D),edge(D,E),node(D,F),node(A,F),edge(E,G),node(E,H),node(B,H), !.
cyclic(A,neg) :- edge(A,B),edge(B,C),edge(C,D),edge(D,E),node(D,F),node(A,F),edge(E,G), !.
cyclic(A,neg) :- edge(A,B),edge(B,C),edge(C,D),edge(D,E),node(D,F),node(A,F), !.
cyclic(A,neg) :- edge(A,B),edge(B,C),edge(C,D),edge(D,E), !.
cyclic(A,neg) :- edge(A,B),edge(B,C), !.
cyclic(A,neg).
