eq_atomtype(A,B) :- A=B.
eq_bondtype(A,B) :- A=B.
eq_charge(A,B) :- A=B.
cycle3(Id,A,B,C) :- atom(Id,A,_,_),atom(Id,B,_,_),atom(Id,C,_,_),bond(Id,A,B,_),bond(Id,B,C,_),bond(Id,C,A,_).
cycle4(Id,A,B,C,D) :- atom(Id,A,_,_),atom(Id,B,_,_),atom(Id,C,_,_),atom(Id,D,_,_),bond(Id,A,B,_),bond(Id,B,C,_),bond(Id,C,D,_),bond(Id,D,A,_).

