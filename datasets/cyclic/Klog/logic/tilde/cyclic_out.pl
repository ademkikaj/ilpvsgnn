:- style_check(-singleton).
cyclic(A,pos) :- edge(A,B),edge(B,C),edge(C,D),edge(D,E),edge(E,F),edge(F,G),edge(G,H),node(G,I),node(A,I),edge(H,J),edge(J,K),node(J,L),node(C,L), !.
cyclic(A,neg) :- edge(A,B),edge(B,C),edge(C,D),edge(D,E),edge(E,F),edge(F,G),edge(G,H),node(G,I),node(A,I),edge(H,J),edge(J,K), !.
cyclic(A,neg) :- edge(A,B),edge(B,C),edge(C,D),edge(D,E),edge(E,F),edge(F,G),edge(G,H),node(G,I),node(A,I), !.
cyclic(A,neg) :- edge(A,B),edge(B,C),edge(C,D),edge(D,E),edge(E,F),edge(F,G),edge(G,H), !.
cyclic(A,neg) :- edge(A,B),edge(B,C),edge(C,D),edge(D,E),edge(E,F), !.
cyclic(A,neg) :- edge(A,B),edge(B,C),edge(C,D), !.
cyclic(A,neg) :- edge(A,B), !.
cyclic(A,neg).
