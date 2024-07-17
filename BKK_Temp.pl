:- use_module('metagol').
:- multifile body_pred/1.
:- multifile head_pred/1.
%%%%%%%%%% tell metagol to use the BK %%%%%%%%%%
body_pred(square/2).
body_pred(in/3).
body_pred(triangle/2).
body_pred(circle/2).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%% metarules %%%%%%%%%%%%%%%%%%%%
metarule([P,Q], [P,A],[[Q,A]]).
metarule([P,Q,R], [P,A], [[Q,A],[R,A]]).
metarule([P,Q,R], [P,A], [[Q,A,B],[R,B]]).
metarule([P,Q], [P,A,B], [[Q,A,B]]).
metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,B]]).
metarule([P,Q,X], [P,A], [[Q,A,X]]).
metarule([P,Q,X], [P,A,B], [[Q,A,B,X]]).
metarule([P,Q], [P,A,B], [[Q,A,B]]).
metarule([P,Q], [P,A,B], [[Q,B,A]]).
metarule([P,Q,R], [P,A,B], [[Q,A],[R,A,B]]).
metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,B]]).
metarule([P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]]).
%%%%%%%%%%%%%% background knowledge %%%%%%%%%%%%%%
square(2,o3).
square(3,o1).
square(3,o2).
square(4,o1).
square(4,o2).
square(5,o6).
square(6,o2).
square(6,o3).
square(6,o4).
square(7,o3).
square(8,o1).
square(8,o2).
square(8,o3).
square(9,o3).
square(10,o1).
square(10,o2).
square(10,o5).
square(10,o7).
square(11,o3).
square(11,o5).
square(11,o7).
square(12,o2).
square(12,o3).
square(13,o2).
square(14,o2).
square(14,o3).
square(14,o6).
square(15,o7).
square(16,o1).
square(16,o2).
square(17,o2).
square(18,o3).
square(18,o6).
square(19,o3).
in(0,o1,o3).
in(0,o2,o3).
in(2,o3,o1).
in(2,o4,o3).
in(2,o1,o4).
in(3,o7,o4).
in(3,o5,o1).
in(3,o4,o5).
in(6,o1,o4).
in(6,o2,o3).
in(7,o1,o3).
in(8,o1,o3).
in(9,o1,o3).
in(9,o2,o3).
in(11,o6,o4).
in(11,o7,o6).
in(11,o4,o5).
in(12,o6,o3).
in(12,o2,o3).
in(12,o7,o6).
in(12,o1,o6).
in(13,o4,o3).
in(13,o3,o4).
in(13,o1,o2).
in(14,o4,o1).
in(14,o1,o3).
in(14,o5,o6).
in(15,o5,o1).
in(15,o3,o1).
in(15,o1,o7).
in(16,o5,o1).
in(16,o4,o3).
in(16,o2,o1).
in(17,o3,o2).
in(17,o4,o3).
in(18,o5,o4).
in(18,o6,o1).
in(18,o7,o6).
in(18,o4,o2).
in(19,o2,o1).
in(19,o4,o3).
in(19,o3,o2).
triangle(0,o3).
triangle(1,o3).
triangle(2,o1).
triangle(3,o3).
triangle(3,o4).
triangle(3,o5).
triangle(3,o6).
triangle(4,o3).
triangle(5,o1).
triangle(5,o2).
triangle(5,o3).
triangle(5,o4).
triangle(7,o2).
triangle(9,o1).
triangle(10,o4).
triangle(10,o6).
triangle(11,o6).
triangle(12,o1).
triangle(12,o7).
triangle(13,o4).
triangle(14,o4).
triangle(15,o2).
triangle(15,o4).
triangle(15,o5).
triangle(15,o6).
triangle(16,o4).
triangle(16,o6).
triangle(17,o1).
triangle(17,o4).
triangle(18,o1).
triangle(18,o4).
triangle(19,o2).
circle(0,o1).
circle(0,o2).
circle(1,o1).
circle(1,o2).
circle(2,o2).
circle(2,o4).
circle(3,o7).
circle(5,o5).
circle(5,o7).
circle(6,o1).
circle(7,o1).
circle(7,o4).
circle(8,o4).
circle(8,o5).
circle(9,o2).
circle(10,o3).
circle(11,o1).
circle(11,o2).
circle(11,o4).
circle(12,o4).
circle(12,o5).
circle(12,o6).
circle(13,o1).
circle(13,o3).
circle(14,o1).
circle(14,o5).
circle(15,o1).
circle(15,o3).
circle(16,o3).
circle(16,o5).
circle(17,o3).
circle(18,o2).
circle(18,o5).
circle(18,o7).
circle(19,o1).
circle(19,o4).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

