:- use_module(library(aleph)).
:- if(current_predicate(use_rendering/1)).
:- use_rendering(prolog).
:- endif.
:- aleph.
:- style_check(-discontiguous).
:- aleph_set(clauselength,10).
:- aleph_set(verbosity,1).
:- aleph_set(minpos,3).
:- aleph_set(nodes,20000).
:- modeh(1,bongard(+id)).
:- modeb(*,square(+id, -object)).
:- modeb(*,circle(+id, -object)).
:- modeb(*,triangle(+id, -object)).
:- modeb(*,in(+id, -object)).
:- modeb(*,edge(+id, -object, -object)).
:- determination(bongard/1,square/2).
:- determination(bongard/1,circle/2).
:- determination(bongard/1,triangle/2).
:- determination(bongard/1,in/2).
:- determination(bongard/1,edge/3).
