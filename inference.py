import os

load_model="Linear_test.pkl"   #name of saved model(use .pkl as file extenstion) (type: string)
infe_model='vgg_cifar10_inference' #vgg_cifar10_inference or resnet_inference
wbit="1"                       #precision of weight (type: int)
abit="1"                       #precision of activation (type: int)
complementary='false'          #Negative activation or not (type: bool)
Separation="True"              #Do weight separation or not (type: bool)
FT="false"                     #Do fine tune or not (type: bool)

RH="2e6"                       #High resistance(unit:ohm) (type: float)
RL="1e6"                       #Low resistance(unit:ohm) (type: float)
std="0.000001"                 #Standard deviation of device variation on Log-normal distribution(type: float)
implementation="MLC"           #type "MLC" or "digital" or "analog"
Max_SA_current="300e-6"
node="0.04"                    #technology node(unit: um)(type:float)
unit_cell="50"                 #unit cell size

save_dir="results"             #saved folder of training accuracy (type: string)

#Assume 1V read voltage

if implementation == "MLC":
    cmd = "python3 main_inferen.py --load_model {0} --model {1} --w_bit {2} --a_bit {3} --comp {4} --sep {5} --finetune {6} --RH {7} --RL {8} --std {9} --Imax {10} --node {11} --cell {12} --save {13}"\
        .format(load_model,infe_model,wbit,abit,complementary,Separation,FT,RH,RL,std,Max_SA_current,node,unit_cell,save_dir)

    os.system(cmd)

if implementation == "digital":
    cmd = "python3 main_inferen_digi.py --load_model {0} --model {1} --w_bit {2} --a_bit {3} --comp {4} --sep {5} --finetune {6} --RH {7} --RL {8} --std {9} --Imax {10} --node {11} --cell {12} --save {13}"\
        .format(load_model,infe_model,wbit,abit,complementary,Separation,FT,RH,RL,std,Max_SA_current,node,unit_cell,save_dir)

    os.system(cmd)

if implementation == "analog":
    cmd = "python3 main_inferen_ana.py --load_model {0} --model {1} --w_bit {2} --a_bit {3} --comp {4} --sep {5} --finetune {6} --RH {7} --RL {8} --std {9} --Imax {10} --node {11} --cell {12} --save {13}"\
        .format(load_model,infe_model,wbit,abit,complementary,Separation,FT,RH,RL,std,Max_SA_current,node,unit_cell,save_dir)

    os.system(cmd)