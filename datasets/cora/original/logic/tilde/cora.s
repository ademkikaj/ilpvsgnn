output_options([c45,c45c,c45e,lp,prolog]).
use_packs(ilp).
predict(cora(+B,-C)).
warmode(content(+paper_id,+-word_cited_id)).
warmode(cites(+-paper_id,+-paper_id)).
write_predictions([testing, distribution]).
combination_rule([product, sum]).
execute(t).
execute(q).
