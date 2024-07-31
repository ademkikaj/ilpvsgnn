:- style_check(-singleton).
sameGen(A,B,pos) :- same_gen(B,C),same_gen(A,D),parent(A,E),parent(C,F),parent(B,G),parent(E,H),parent(F,I), !.
sameGen(A,B,neg) :- same_gen(B,C),same_gen(A,D),parent(A,E),parent(C,F),parent(B,G),parent(E,H), !.
sameGen(A,B,pos) :- same_gen(B,C),same_gen(A,D),parent(A,E),parent(C,F),parent(B,G), !.
sameGen(A,B,pos) :- same_gen(B,C),same_gen(A,D),parent(A,E),parent(C,F), !.
sameGen(A,B,neg) :- same_gen(B,C),same_gen(A,D),parent(A,E), !.
sameGen(A,B,neg) :- same_gen(B,C),same_gen(A,D),parent(B,E),parent(D,F),parent(E,G), !.
sameGen(A,B,pos) :- same_gen(B,C),same_gen(A,D),parent(B,E),parent(D,F), !.
sameGen(A,B,neg) :- same_gen(B,C),same_gen(A,D),parent(B,E), !.
sameGen(A,B,pos) :- same_gen(B,C),same_gen(A,D),parent(D,E),parent(C,F), !.
sameGen(A,B,neg) :- same_gen(B,C),same_gen(A,D),parent(D,E), !.
sameGen(A,B,pos) :- same_gen(B,C),same_gen(A,D), !.
sameGen(A,B,neg) :- same_gen(B,C), !.
sameGen(A,B,neg).
