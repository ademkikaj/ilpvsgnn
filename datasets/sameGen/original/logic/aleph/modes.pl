:- aleph_set(i,2).
:- aleph_set(verbosity,1).
:- aleph_set(clauselength,6).
:- aleph_set(minacc,0.2):- aleph_set(minpos,2).
:- aleph_set(nodes,20000).
:- aleph_set(noise,5).
:- aleph_set(c,3).
:- modeh(1,sameGen(+name,+name)).
:- modeb(*,same_gen(+name,-name)).
:- modeb(*,person(+name)).
:- modeb(*,parent(+name,-name)).
:- determination(sameGen/2,same_gen/2).
:- determination(sameGen/2,person/1).
:- determination(sameGen/2,parent/2).
