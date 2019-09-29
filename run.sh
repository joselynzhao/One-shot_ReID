# run DukeMTMC-VideoReID
python3.6 run.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_EF_30_q_0.4/ --EF 30 --q 0.4 --mode Dissimilarity --max_frames 400 --resume logs/DukeMTMC_VideoReID_EF_30_q_0.4/
#python3.6 run.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_EF_20_q_1/ --EF 20 --q 1.5 --mode Dissimilarity --max_frames 100 --resume logs/DukeMTMC_VideoReID_EF_20_q_1/

# run mars
#python3.6 run.py --dataset mars --logs_dir logs/mars_EF_10/ --EF 10 --mode Dissimilarity --max_frames 100

# if you need to resume 
# python3 run.py --dataset mars --logs_dir logs/mars_EF_10/ --EF 10 --mode Dissimilarity --max_frames 900 --resume logs/mars_EF_10/ 

