same_row(A,B) :- white_rook(_,_,A,_),white_king(_,_,B,_),A=:=B.
same_col(A,B) :- white_rook(_,_,_,A),black_king(_,_,_,B),A=:=B.
