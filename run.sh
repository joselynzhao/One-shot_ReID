# run DukeMTMC-VideoReID
#python3.6 run2.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_AP_bs_50/  --mode Dissimilarity --max_frames 400  --bs 50
#python3.6 run.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_EF_20_q_1/ --EF 20 --q 1.5 --mode Dissimilarity --max_frames 100 --resume logs/DukeMTMC_VideoReID_EF_20_q_1/

# run mars
#python3.6 run.py --dataset mars --logs_dir logs/mars_EF_10/ --EF 10 --mode Dissimilarity --max_frames 100

# if you need to resume 
#python3.6 run3.py --dataset mars --logs_dir logs/mars_EF_10_q_1_pro/ --EF 10 --q 1 --mode Dissimilarity --max_frames 100




python3.6 run_supervise2.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_supervise20_step2_EF2/ --EF 2 --q 1 --mode Dissimilarity --max_frames 400
