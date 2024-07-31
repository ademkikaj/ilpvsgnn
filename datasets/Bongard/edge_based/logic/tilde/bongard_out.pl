:- style_check(-singleton).
bongard(A,pos) :- shape(A,triangle,B),in(A,B,C),shape(A,triangle,C), !.
bongard(A,neg) :- shape(A,triangle,B),in(A,B,C),shape(A,circle,C),shape(A,circle,D),in(A,D,E), !.
bongard(A,pos) :- shape(A,triangle,B),in(A,B,C),shape(A,circle,C), !.
bongard(A,pos) :- shape(A,triangle,B),in(A,B,C),shape(A,square,D),in(A,D,E),shape(A,circle,E), !.
bongard(A,neg) :- shape(A,triangle,B),in(A,B,C),shape(A,square,D),in(A,D,E), !.
bongard(A,neg) :- shape(A,triangle,B),in(A,B,C), !.
bongard(A,neg) :- shape(A,circle,B),in(A,B,C), !.
bongard(A,pos) :- shape(A,circle,B), !.
bongard(A,neg).
