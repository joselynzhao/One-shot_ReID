# run DukeMTMC-VideoReID
#python3.6 run2.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_AP_bs_50/  --mode Dissimilarity --max_frames 400  --bs 50
#python3.6 run.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_EF_20_q_1/ --EF 20 --q 1.5 --mode Dissimilarity --max_frames 100 --resume logs/DukeMTMC_VideoReID_EF_20_q_1/

# run mars
#python3.6 run.py --dataset mars --logs_dir logs/mars_EF_10/ --EF 10 --mode Dissimilarity --max_frames 100

# if you need to resume 
#python3.6 run3.py --dataset mars --logs_dir logs/mars_EF_10_q_1_pro/ --EF 10 --q 1 --mode Dissimilarity --max_frames 100

#python3.6 run3.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_yita70JZ_50/  --yita 50  --mode Dissimilarity --max_frames 900 --device 0
#python3.6 run3.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_yita70JZ_100/  --yita 100  --mode Dissimilarity --max_frames 900 --device 1
#python3.6 run3.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_yita70JZ_200/  --yita 200  --mode Dissimilarity --max_frames 900 --device 2
#python3.6 run3.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_yita70JZ_300/  --yita 300  --mode Dissimilarity --max_frames 900 --device 3
python3.6 run3.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_yita70JZ_0/  --yita 0  --mode Dissimilarity --max_frames 900 --device 4  #验证作者的方法
