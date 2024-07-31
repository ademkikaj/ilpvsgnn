:- style_check(-singleton).
train(A,pos):-long(A,B),three_wheels(A,C),roof_closed(A,B), !.

train(A,neg).
