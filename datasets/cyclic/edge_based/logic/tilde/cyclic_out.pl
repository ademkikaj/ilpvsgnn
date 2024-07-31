:- style_check(-singleton).
cyclic(A,pos) :- edge(A,B),instance(B),instance(A),edge(B,C),instance(C), !.
cyclic(A,neg) :- edge(A,B),instance(B),instance(A), !.
cyclic(A,neg) :- edge(A,B),instance(B), !.
cyclic(A,neg).
