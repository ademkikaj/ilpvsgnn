:- style_check(-singleton).
train(A,pos) :- three_wheels(A,B),roof_closed(A,C),long(A,C), !.
train(A,neg) :- three_wheels(A,B),roof_closed(A,C), !.
train(A,neg) :- three_wheels(A,B), !.
train(A,neg).
