# experiments_name : NLVM-b1
#python3.6 nlvm.py --exp_order 0 --percent_vari 1
#python3.6 nlvm.py --exp_order 1 --percent_vari 0.9
#python3.6 nlvm.py --exp_order 2 --percent_vari 0.8
#python3.6 nlvm.py --exp_order 0 --percent_vari 0.8  --exp_name test  --epoch 0



# experiments_name : NLVM-b2
#python3.6 nlvm-b2.py --exp_order 1  --exp_name nlvm-b2   --stop_vari_step 8
#python3.6 nlvm-b2.py --exp_order 2  --exp_name nlvm-b2   --stop_vari_step 7
python3.6 nlvm-b2.py --exp_order 3  --exp_name nlvm-b2   --stop_vari_step 6  --resume Ture
python3.6 nlvm-b2.py --exp_order 4  --exp_name nlvm-b2   --stop_vari_step 5