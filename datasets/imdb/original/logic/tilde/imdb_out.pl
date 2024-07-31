:- style_check(-singleton).
imdb(A,B,pos) :- genre(B,C), !.
imdb(A,B,neg).
