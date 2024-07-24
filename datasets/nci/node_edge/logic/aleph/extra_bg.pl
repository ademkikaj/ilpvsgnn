eq_atom(A,B) :- A=B.
eq_bond(A,B) :- A=B.
%cycle3(Id,A,B,C) :- atom(Id,A,_),atom(Id,B,_),atom(Id,C,_),bond(Id,A,B,_),bond(Id,B,C,_),bond(Id,C,A,_).
