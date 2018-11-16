set   autoscale                       
unset log                              
unset label                            
set xtic auto                          
set ytic auto                          
set title "Mean Trend for Gf2x and Ack Multiply"
set xlabel "Operand Length"
set ylabel "Clock Cycles"
set style line 1 lt 1 lc rgb "black"
set style line 2 lt 1 lc rgb "grey"
set style line 3 lt 1 lc rgb "blue"
set style line 4 lt 1 lc rgb "cyan"
plot  n=1 "Test_file/AckResult.dat" u 1:2 t 'Ack' w linespoints ls n, \
        n=2 "Test_file/AckResult.dat" u 1:($2-sqrt($3)) t 'Ack bot' w linespoints ls n, \
        n=2 "Test_file/AckResult.dat" u 1:($2+sqrt($3))  t 'Ack top' w linespoints ls n, \
        n=3 "Test_file/Gf2xResult.dat" u 1:2 t 'Gf2x' w linespoints ls n, \
        n=4 "Test_file/Gf2xResult.dat" u 1:($2-sqrt($3)) t 'Gf2x bot' w linespoints ls n, \
        n=4 "Test_file/Gf2xResult.dat" u 1:($2+sqrt($3)) t 'Gf2x top' w linespoints ls n, \

