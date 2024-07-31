:- style_check(-singleton).
krk(A,pos):- cell(A,D,C,G),king(G),cell(A,F,C,E),distance(F,D,B),rook(E),black(C),one(B),!.
krk(A,neg).
