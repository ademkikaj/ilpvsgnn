:- aleph_set(i,2).
:- aleph_set(verbosity,1).
:- aleph_set(clauselength,6).
:- aleph_set(nodes,20000).
:- aleph_set(noise,5).
:- aleph_set(c,3).

:- modeh(1,bongard(+puzzle)).

:- modeb(*,square(+puzzle,-object)).
:- modeb(*,circle(+puzzle,-object)).
:- modeb(*,triangle(+puzzle,-object)).

:- modeb(*,in(+puzzle,-object,-object)).

:- determination(bongard/1,square/2).
:- determination(bongard/1,circle/2).
:- determination(bongard/1,triangle/2).
:- determination(bongard/1,in/3).
