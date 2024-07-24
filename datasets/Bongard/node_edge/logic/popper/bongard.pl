:- style_check(-singleton).
bongard(A,pos):- triangle(A,B),edge(A,C,B),triangle(A,C), !.
bongard(A,neg).
