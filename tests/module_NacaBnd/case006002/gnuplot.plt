set xrange [-0.1:1.1]
set yrange [-0.1:0.1]
set size ratio -1
plot 'output.out' using 1:2:(-0.02*$3):(-0.02*$4) with vectors  head  filled lt 2 notitle,\
     'output.out' using 1:2 w p lc 1 notitle

