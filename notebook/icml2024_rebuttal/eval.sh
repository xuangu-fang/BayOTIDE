
for D_s in 20;

do 

for D_t in 3 5 10 20 30 40;

do 

python fast_test.py -dataset="uber" -D_trend=$D_t -D_season=$D_s

done

done