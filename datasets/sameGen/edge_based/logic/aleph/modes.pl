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
:- modeb(*,parent(+id, -node_id, -name,-name)).
:- modeb(*,person(+id, -node_id, -name)).
:- modeb(*,same_gen(+id, -node_id, -node_id,-name)).
:- modeb(*,edge(+id, -node_id, -node_id)).
:- modeb(*,instance(+id, -node_id)).
:- determination(sameGen/3,parent/3).
:- determination(sameGen/3,person/3).
:- determination(sameGen/3,same_gen/3).
:- determination(sameGen/3,edge/3).
:- determination(sameGen/3,instance/2).
