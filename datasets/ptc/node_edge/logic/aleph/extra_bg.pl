eq_atom(A,B) :- A=B.
eq_bond(A,B) :- A=B.
cycle3(Id,A,B,C) :- atom(Id,A,_),atom(Id,B,_),atom(Id,C,_),connected(Id,A,B,_),connected(Id,B,C,_),connected(Id,C,A,_).
cycle4(Id,A,B,C,D) :- atom(Id,A,_),atom(Id,B,_),atom(Id,C,_),atom(Id,D,_),connected(Id,A,B,_),connected(Id,B,C,_),connected(Id,C,D,_),connected(Id,D,A,_).
cycle5(Id,A,B,C,D,E) :- atom(Id,A,_),atom(Id,B,_),atom(Id,C,_),atom(Id,D,_),atom(Id,E,_),connected(Id,A,B,_),connected(Id,B,C,_),connected(Id,C,D,_),connected(Id,D,E,_),connected(Id,E,A,_).

