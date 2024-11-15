#!bin/bash

# OfficeHome
# for seed in 1 2; do
#     python main.py --dataset OfficeHome --source Art --target Clipart --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Art --target Product --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Art --target Real_World --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Clipart --target Art --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Clipart --target Product --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Clipart --target Real_World --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Product --target Art --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Product --target Clipart --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Product --target Real_World --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Real_World --target Art --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Real_World --target Clipart --method retrain --fast_train --seed "$seed"
#     python main.py --dataset OfficeHome --source Real_World --target Product --method retrain --fast_train --seed "$seed"
# done

for seed in 2; do
    python main.py --dataset OfficeHome --source Art --target Clipart --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Product --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Real_World --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Art --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Product --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Real_World --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Art --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Clipart --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Real_World --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Art --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Clipart --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Product --method unsir --fast_train --seed "$seed"
done

for seed in 3; do
    python main.py --dataset OfficeHome --source Art --target Clipart --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Product --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Real_World --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Art --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Product --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Real_World --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Art --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Clipart --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Real_World --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Art --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Clipart --method minimax --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Product --method minimax --fast_train --seed "$seed"
    

    python main.py --dataset OfficeHome --source Art --target Clipart --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Product --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Real_World --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Art --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Product --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Real_World --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Art --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Clipart --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Real_World --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Art --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Clipart --method finetune --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Product --method finetune --fast_train --seed "$seed"
    
    python main.py --dataset OfficeHome --source Art --target Clipart --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Product --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Real_World --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Art --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Product --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Real_World --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Art --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Clipart --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Real_World --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Art --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Clipart --method original --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Product --method original --fast_train --seed "$seed"
    
    python main.py --dataset OfficeHome --source Art --target Clipart --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Product --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Real_World --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Art --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Product --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Real_World --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Art --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Clipart --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Real_World --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Art --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Clipart --method retrain --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Product --method retrain --fast_train --seed "$seed"
    
    python main.py --dataset OfficeHome --source Art --target Clipart --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Product --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Art --target Real_World --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Art --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Product --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Clipart --target Real_World --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Art --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Clipart --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Product --target Real_World --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Art --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Clipart --method unsir --fast_train --seed "$seed"
    python main.py --dataset OfficeHome --source Real_World --target Product --method unsir --fast_train --seed "$seed"
done

# Office31
  
# python main.py --dataset Office31 --source amazon --target dslr --method minimax
# python main.py --dataset Office31 --source amazon --target webcam --method minimax
# python main.py --dataset Office31 --source dslr --target amazon --method minimax
# python main.py --dataset Office31 --source dslr --target webcam --method minimax
# python main.py --dataset Office31 --source webcam --target amazon --method minimax
# python main.py --dataset Office31 --source webcam --target dslr --method minimax
  
# python main.py --dataset Office31 --source amazon --target dslr --method finetune
# python main.py --dataset Office31 --source amazon --target webcam --method finetune
# python main.py --dataset Office31 --source dslr --target amazon --method finetune
# python main.py --dataset Office31 --source dslr --target webcam --method finetune
# python main.py --dataset Office31 --source webcam --target amazon --method finetune
# python main.py --dataset Office31 --source webcam --target dslr --method finetune
  
# python main.py --dataset Office31 --source amazon --target dslr --method original
# python main.py --dataset Office31 --source amazon --target webcam --method original
# python main.py --dataset Office31 --source dslr --target amazon --method original
# python main.py --dataset Office31 --source dslr --target webcam --method original
# python main.py --dataset Office31 --source webcam --target amazon --method original
# python main.py --dataset Office31 --source webcam --target dslr --method original
  
# python main.py --dataset Office31 --source amazon --target dslr --method retrain
# python main.py --dataset Office31 --source amazon --target webcam --method retrain
# python main.py --dataset Office31 --source dslr --target amazon --method retrain
# python main.py --dataset Office31 --source dslr --target webcam --method retrain
# python main.py --dataset Office31 --source webcam --target amazon --method retrain
# python main.py --dataset Office31 --source webcam --target dslr --method retrain
  
# python main.py --dataset Office31 --source amazon --target dslr --method unsir
# python main.py --dataset Office31 --source amazon --target webcam --method unsir
# python main.py --dataset Office31 --source dslr --target amazon --method unsir
# python main.py --dataset Office31 --source dslr --target webcam --method unsir
# python main.py --dataset Office31 --source webcam --target amazon --method unsir
# python main.py --dataset Office31 --source webcam --target dslr --method unsir

# DomainNet S→P C→S P→C P→R R→S R→C R→P 

# python main.py --dataset DomainNet --source sketch --target painting --method minimax --batch 24
# python main.py --dataset DomainNet --source clipart --target sketch --method minimax --batch 24
# python main.py --dataset DomainNet --source painting --target clipart --method minimax --batch 24
# python main.py --dataset DomainNet --source painting --target real --method minimax --batch 24
# python main.py --dataset DomainNet --source real --target sketch --method minimax --batch 24
# python main.py --dataset DomainNet --source real --target clipart --method minimax --batch 24
# python main.py --dataset DomainNet --source real --target painting --method minimax --batch 24

# python main.py --dataset DomainNet --source sketch --target painting --method original --fast_train --epochs 20
# python main.py --dataset DomainNet --source clipart --target sketch --method original --fast_train --epochs 20
# python main.py --dataset DomainNet --source painting --target clipart --method original --fast_train --epochs 20
# python main.py --dataset DomainNet --source painting --target real --method original --fast_train --epochs 20
# python main.py --dataset DomainNet --source real --target sketch --method original --fast_train --epochs 20
# python main.py --dataset DomainNet --source real --target clipart --method original --fast_train --epochs 20
# python main.py --dataset DomainNet --source real --target painting --method original --fast_train --epochs 20

# python main.py --dataset DomainNet --source sketch --target painting --method retrain --fast_train --epochs 20
# python main.py --dataset DomainNet --source clipart --target sketch --method retrain --fast_train --epochs 20
# python main.py --dataset DomainNet --source painting --target clipart --method retrain --fast_train --epochs 20
# python main.py --dataset DomainNet --source painting --target real --method retrain --fast_train --epochs 20
# python main.py --dataset DomainNet --source real --target sketch --method retrain --fast_train --epochs 20
# python main.py --dataset DomainNet --source real --target clipart --method retrain --fast_train --epochs 20
# python main.py --dataset DomainNet --source real --target painting --method retrain --fast_train --epochs 20

# python main.py --dataset DomainNet --source sketch --target painting --method finetune --fast_train --epochs 20
# python main.py --dataset DomainNet --source clipart --target sketch --method finetune --fast_train --epochs 20
# python main.py --dataset DomainNet --source painting --target clipart --method finetune --fast_train --epochs 20
# python main.py --dataset DomainNet --source painting --target real --method finetune --fast_train --epochs 20
# python main.py --dataset DomainNet --source real --target sketch --method finetune --fast_train --epochs 20
# python main.py --dataset DomainNet --source real --target clipart --method finetune --fast_train --epochs 20
# python main.py --dataset DomainNet --source real --target painting --method finetune --fast_train --epochs 20

# OfficeHome Continual

# python main.py --dataset OfficeHome --source Art --target Clipart --method minimax_continual 
# python main.py --dataset OfficeHome --source Art --target Product --method minimax_continual 
# python main.py --dataset OfficeHome --source Art --target Real_World --method minimax_continual 
# python main.py --dataset OfficeHome --source Clipart --target Art --method minimax_continual 
# python main.py --dataset OfficeHome --source Clipart --target Product --method minimax_continual 
# python main.py --dataset OfficeHome --source Clipart --target Real_World --method minimax_continual 
# python main.py --dataset OfficeHome --source Product --target Art --method minimax_continual 
# python main.py --dataset OfficeHome --source Product --target Clipart --method minimax_continual 
# python main.py --dataset OfficeHome --source Product --target Real_World --method minimax_continual 
# python main.py --dataset OfficeHome --source Real_World --target Art --method minimax_continual 
# python main.py --dataset OfficeHome --source Real_World --target Clipart --method minimax_continual 
# python main.py --dataset OfficeHome --source Real_World --target Product --method minimax_continual 
  
# Ablation OfficeHome Different Alpha

# for alpha in 1.0 5.0 10.0 15.0; do
#     echo "$alpha"
#     printf "Using Alpha = $alpha\n" >> log.txt
#     python main.py --dataset OfficeHome --source Art --target Clipart --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Art --target Product --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Art --target Real_World --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Clipart --target Art --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Clipart --target Product --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Clipart --target Real_World --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Product --target Art --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Product --target Clipart --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Product --target Real_World --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Real_World --target Art --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Real_World --target Clipart --method minimax --alpha "$alpha" --fast_train
#     python main.py --dataset OfficeHome --source Real_World --target Product --method minimax --alpha "$alpha" --fast_train
# done

# Ablation OfficeHome Different Num Adv

# for num_adv in 1 4 8 16; do
#     echo "$num_adv"
#     printf "Using $num_adv Adv\n" >> log.txt
#     python main.py --dataset OfficeHome --source Art --target Clipart --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Art --target Product --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Art --target Real_World --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Clipart --target Art --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Clipart --target Product --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Clipart --target Real_World --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Product --target Art --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Product --target Clipart --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Product --target Real_World --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Real_World --target Art --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Real_World --target Clipart --method minimax --num_adv "$num_adv" --fast_train
#     python main.py --dataset OfficeHome --source Real_World --target Product --method minimax --num_adv "$num_adv" --fast_train
# done