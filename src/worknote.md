#### ssh connection

ssh ml-jsha@130.83.185.153
ssh ml-jsha@130.83.185.155
ssh ml-jsha@130.83.185.145
ssh ml-jsha@130.83.185.147
ssh ml-jsha@130.83.42.209
ssh ml-jsha@130.83.42.211

###### Build docker
docker build -t ml-sha/alphailp_docker .

###### Run docker
docker run --gpus all -it --rm ml-sha/alphailp_docker

###### Run experiment: nearby

git clone https://github.com/akweury/alpha_ilp_origional.git

pip install opencv-python==4.5.5.64

###### Exp: closeby
python3 src/train.py --dataset-type kandinsky --dataset closeby-learn --batch-size 1 --batch-size-train 20 --n-beam 5 --t-beam 4 --epochs 101 --no-pi --pi_epochs 2  --device 5
python3 src/train.py --dataset-type kandinsky --dataset closeby-learn --batch-size 1 --batch-size-train 20 --n-beam 5 --t-beam 4 --epochs 101 --pi_epochs 2  --device 6

###### Exp: twopairs
python3 src/train.py --dataset-type kandinsky --dataset twopairs --batch-size 1 --batch-size-train 20 --n-beam 5 --t-beam 5 --epochs 101 --no-pi --pi_epochs 2  --device 6
python3 src/train.py --dataset-type kandinsky --dataset twopairs --batch-size 1 --batch-size-train 20 --n-beam 5 --t-beam 5 --epochs 101 --pi_epochs 2  --device 7


###### Exp: red triangle
python3 src/train.py --dataset-type kandinsky --dataset red-triangle --batch-size 1 --batch-size-train 10 --n-beam 5 --t-beam 6 --epochs 101  --device 7 --n-obj 2
python3 src/train.py --dataset-type kandinsky --dataset red-triangle --batch-size 1 --n-beam 5 --t-beam 4 --device 2 --epochs 101 --pi_epochs 3
python3 src/train.py --dataset-type kandinsky --dataset red-triangle --batch-size 1 --n-beam 5 --t-beam 4 --device 1 --epochs 101
python3 src/train.py --dataset-type kandinsky --dataset red-triangle --batch-size 1 --n-beam 5 --t-beam 4 --device 1 --epochs 5

###### Exp: online-pair
python3 src/train.py --dataset online-pair --batch-size-train 10  --t-beam 8 --pi_epochs 3  --n-obj 3 --device 6
python3 src/train.py --dataset online-pair --t-beam 4 --pi_epochs 3 --n-obj 5 --device 7 


###### Exp: online-5
python3 src/train.py --dataset online-5-pair --batch-size-train 5  --t-beam 8 --pi_epochs 3  --n-obj 5 --sn_th 0.9 --device 6
python3 src/train.py --dataset online-5-pair  --t-beam 8  --n-obj 5 --device 6
 

