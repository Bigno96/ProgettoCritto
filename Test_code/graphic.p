set   autoscale                       
unset log                              
unset label                            
set xtic auto                          
set ytic auto                          
set title "Mean Trend for Gf2x and Ack Multiply"
set xlabel "Operand Length"
set ylabel "Clock Cycles"
plot    "Test_file/AckResult.dat" u 1:2 t 'Ack' w linespoints , "Test_file/Gf2xResult.dat" u 1:2 t 'Gf2x' w linespoint
