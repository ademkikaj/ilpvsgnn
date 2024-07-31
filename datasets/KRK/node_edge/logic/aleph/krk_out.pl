:- style_check(-singleton).
krk(A,pos):-white_king(A,B,C,D),white_rook(A,E,C,D), !.
krk(F,pos):-white_rook(F,G,H,I),black_king(F,J,H,K), !.
krk(L,pos):-white_rook(L,M,N,O),black_king(L,P,Q,O), !.

krk(A,neg).
