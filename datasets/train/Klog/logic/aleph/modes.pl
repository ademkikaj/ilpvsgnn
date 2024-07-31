:- use_module(library(aleph)).
:- if(current_predicate(use_rendering/1)).
:- use_rendering(prolog).
:- endif.
:- aleph.
:- style_check(-discontiguous).
:- aleph_set(verbosity,1).
:- aleph_set(minpos,2).
:- aleph_set(nodes,100000).
:- modeh(1,train(+id)).
:- modeb(*,has_car(+id, -car_id)).
:- modeb(*,has_load(+id, -load_id)).
:- modeb(*,short(+id, -car_id)).
:- modeb(*,long(+id, -car_id)).
:- modeb(*,two_wheels(+id, -car_id)).
:- modeb(*,three_wheels(+id, -car_id)).
:- modeb(*,roof_open(+id, -car_id)).
:- modeb(*,roof_closed(+id, -car_id)).
:- modeb(*,zero_load(+id, -load_id)).
:- modeb(*,one_load(+id, -load_id)).
:- modeb(*,two_load(+id, -load_id)).
:- modeb(*,three_load(+id, -load_id)).
:- modeb(*,circle(+id, -load_id)).
:- modeb(*,triangle(+id, -load_id)).
:- modeb(*,edge(+id, -object, -object)).
:- determination(train/1,has_car/2).
:- determination(train/1,has_load/2).
:- determination(train/1,short/2).
:- determination(train/1,long/2).
:- determination(train/1,two_wheels/2).
:- determination(train/1,three_wheels/2).
:- determination(train/1,roof_open/2).
:- determination(train/1,roof_closed/2).
:- determination(train/1,zero_load/2).
:- determination(train/1,one_load/2).
:- determination(train/1,two_load/2).
:- determination(train/1,three_load).
:- determination(train/1,circle/2).
:- determination(train/1,triangle/2).
:- determination(train/1,edge/3).
