same_x(A,B) :- A == B.
same_y(A,B) :- A == B.

distance(X1,Y1,X2,Y2,D) :- D is sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2)).
one(1).

