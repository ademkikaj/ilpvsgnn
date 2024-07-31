:- style_check(-singleton).
cora(A,neg) :- cites(A,A), !.
cora(A,pos) :- cites(A,B), !.
cora(A,pos) :- cites(B,A),content(B,C), !.
cora(A,neg) :- cites(B,A), !.
cora(A,neg).
