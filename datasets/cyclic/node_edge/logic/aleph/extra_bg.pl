%cycle_4(Id,A,B,C,D) :- node(Id,A,_),node(Id,B,_),node(Id,C,_),node(Id,D,_),edge(Id,A,B),edge(Id,B,C),edge(Id,C,D),edge(Id,D,A).
red(red).
green(green).

