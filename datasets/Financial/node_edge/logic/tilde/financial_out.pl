:- style_check(-singleton).
financial(A,neg) :- trans(A,B,C,D,E,F,G),eq_trans_char(G,sankc_urok),card(A,H,I,J), !.
financial(A,pos) :- trans(A,B,C,D,E,F,G),eq_trans_char(G,sankc_urok), !.
financial(A,neg).
