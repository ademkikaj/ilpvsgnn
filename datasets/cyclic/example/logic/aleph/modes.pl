

:- aleph_set(verbosity, 0).
:- aleph_set(i,6).
:- aleph_set(clauselength,5).
:- aleph_set(minpos,2).
:- aleph_set(nodes,30000).

:- modeb(*,edge(+node,-node)).
:- modeb(*,colour(+node,-colour)).
:- modeb(*,red(+colour)).
:- modeb(*,green(+colour)).

:- modeh(*,f(+node)).

:- determination(f/1,edge/2).
:- determination(f/1,colour/2).
:- determination(f/1,red/1).
:- determination(f/1,green/1).
:- determination(f/1,f/1).