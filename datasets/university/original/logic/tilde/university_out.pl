:- style_check(-singleton).
university(A,neg,neg) :- course(B,C,D), !.
university(A,pos,pos).
