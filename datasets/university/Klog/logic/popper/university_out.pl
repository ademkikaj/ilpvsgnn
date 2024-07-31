:- style_check(-singleton).
university(A,pos):- edge(A,B,C),edge(A,C,B), !.
university(A,neg).
