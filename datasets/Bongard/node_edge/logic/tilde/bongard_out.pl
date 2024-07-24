:- style_check(-singleton).
bongard(A,pos) :- triangle(A,B),edge(A,B,C),triangle(A,C), !.
bongard(A,neg) :- triangle(A,B),edge(A,B,C),circle(A,C),circle(A,D),edge(A,D,E), !.
bongard(A,pos) :- triangle(A,B),edge(A,B,C),circle(A,C), !.
bongard(A,pos) :- triangle(A,B),edge(A,B,C),square(A,D),edge(A,D,E),circle(A,E), !.
bongard(A,neg) :- triangle(A,B),edge(A,B,C),square(A,D),edge(A,D,E), !.
bongard(A,neg) :- triangle(A,B),edge(A,B,C), !.
bongard(A,neg) :- circle(A,B),edge(A,B,C), !.
bongard(A,pos) :- circle(A,B), !.
bongard(A,neg).
