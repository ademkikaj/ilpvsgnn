max_vars(8).
max_body(6).
head_pred(train,1).
body_pred(has_car,2).
body_pred(has_load,3).
body_pred(short,2).
body_pred(long,2).
body_pred(two_wheels,2).
body_pred(three_wheels,2).
body_pred(roof_open,2).
body_pred(roof_closed,2).
body_pred(zero_load,2).
body_pred(one_load,2).
body_pred(two_load,2).
body_pred(three_load,2).
body_pred(circle,2).
body_pred(triangle,2).
type(train,(id,)).
type(has_car,(id,car_id)).
type(has_load,(id,car_id,load_id)).
type(short,(id,car_id)).
type(long,(id,car_id)).
type(two_wheels,(id,car_id)).
type(three_wheels,(id,car_id)).
type(roof_open,(id,car_id)).
type(roof_closed,(id,car_id)).
type(zero_load,(id,load_id)).
type(one_load,(id,load_id)).
type(two_load,(id,load_id)).
type(three_load,(id,load_id)).
type(circle,(id,load_id)).
type(triangle,(id,load_id)).
