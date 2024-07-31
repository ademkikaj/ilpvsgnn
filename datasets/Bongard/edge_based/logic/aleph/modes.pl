:- use_module(library(aleph)).
:- if(current_predicate(use_rendering/1)).
:- use_rendering(prolog).
:- endif.
:- aleph.
:- style_check(-discontiguous).
:- aleph_set(verbosity,1).
:- aleph_set(minpos,2).
:- aleph_set(nodes,100000).
:- modeh(1,bongard(+id)).
:- modeb(1,square(+constant)).
:- modeb(1,circle(+constant)).
:- modeb(1,triangle(+constant)).
:- modeb(*,shape(+id, -constant, -object)).
:- modeb(*,in(+id, -object, -object)).
:- modeb(*,instance(+id, -object)).
:- determination(bongard/1,shape/3).
:- determination(bongard/1,in/3).
:- determination(bongard/1,square/1).
:- determination(bongard/1,circle/1).
:- determination(bongard/1,triangle/1).
:- determination(bongard/1,instance/2).
