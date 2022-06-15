import os
wbit="1"                        #precision of weight   
abit="1"                        #precision of activation
complementary='false'           #Negative activation or not
train_model='vgg_cifar10_train' #vgg_cifar10_train or resnet_train
save_dir="results"              #saved folder of training accuracy
save_model="Linear_test.pkl"    #name of saved model(use .pkl as file extenstion)


cmd = "python3 main_train.py --model {0} --w_bit {1} --a_bit {2} --comp {3} --save {4} --save_model {5}"\
    .format(train_model,wbit,abit,complementary,save_dir,save_model)

os.system(cmd)