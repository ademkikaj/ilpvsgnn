:- style_check(-singleton).
bongard(A,pos):-triangle(A,B),triangle(A,C),edge(A,B,C), !.

bongard(A,neg).
