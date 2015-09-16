set xrange [-0.1:1.1]
set yrange [-0.1:0.1]
set size ratio -1
plot 'output.out' using 1:2 w p pt 1 lc 1 title "l_vertex",\
     'output.out' using 3:4 w p pt 2 lc 2 title "r_vertex",\
     'output.out' using 5:6 w p pt 6 lc 3 title "new_vertex" \

