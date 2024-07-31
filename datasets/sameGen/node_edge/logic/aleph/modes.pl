:- use_module(library(aleph)).
:- if(current_predicate(use_rendering/1)).
:- use_rendering(prolog).
:- endif.
:- aleph.
:- style_check(-discontiguous).
:- aleph_set(clauselength,5).
:- aleph_set(i,6).
:- aleph_set(verbosity,1).
:- aleph_set(minpos,5).
:- aleph_set(nodes,30000).
:- modeh(1,sameGen(+id,+name,+name)).
:- modeb(*,parent(+id, -name, -name)).
:- modeb(*,person(+id, -name)).
:- modeb(*,same_gen(+id, -name, -name)).
:- determination(sameGen/3,parent/3).
:- determination(sameGen/3,person/2).
:- determination(sameGen/3,same_gen/3).
