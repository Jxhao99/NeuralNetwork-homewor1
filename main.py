import pandas as pd
from utils import train,test
from model import NeuralNetwork
from load_mnist import load_mnist_datasets
from visualize.visualize import visualize,save_pic_data
import click

input_size = 28*28
output_size = 10
lr_decay = 1
batch_size = 20

train_set, val_data, test_data = load_mnist_datasets('data/mnist.pkl.gz')

X_train = [train_set[0][k:k+batch_size] for k in range(0, len(train_set[0]), batch_size)]
y_train = [train_set[1][k:k+batch_size] for k in range(0, len(train_set[1]), batch_size)]
X_test,y_test = test_data[0],test_data[1]
train_data = [X_train,y_train]
print("Loading dataset success!")
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help']) # -h 生效
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-search', is_flag=True, help="是否参数搜索")
@click.option('-load', is_flag=True, help="是否直接加载模型")
@click.option('-visual', is_flag=True, help="是否可视化")
@click.option("--hidden_size", type=int, default=100)
@click.option("--lr", type=float, default=0.033)
@click.option("--reg", type=float, default=0)
@click.option("--epochs", type=int, default=80)

def main(search,load,visual,hidden_size,lr,reg,epochs):
    if search:
        ##固定hidden_size=40,reg=0,epochs=80  在[0.01,0.03]对lr进行遍历
        ##结果[0.9639, 0.9624, 0.9665, 0.9657, 0.9654, 0.9665, 0.9654, 0.9666, 0.9682, 0.9675]
        ##对[0.025,0.033]遍历  [0.9679, 0.9676, 0.9689, 0.9667, 0.9707]   0.033最佳

        ###对lr = 0.03，batch_sizes = [20,40,80]遍历  batch_size越大越好  80时验证集，测试集上均为97.2%+
        lrs = [0.025+0.002*i for i in range(5)]
        hidden_sizes = [40,60,80,100]
        regs = [0+0.001*i for i in range(5)]
        parameter = [0,0,0]
        best_accuracy = 0
        for Lr in lrs:
            for Hidden_size in hidden_sizes:
                for Reg in regs:
                    print(f"lr: {Lr}batch_size: {Hidden_size},reg: {Reg}")
                    model = NeuralNetwork(input_size,Hidden_size,output_size)
                    curr_accuracy,train_losses,validate_losses,accuracies = train(model,train_data,val_data,epochs,Lr,lr_decay,Reg)
                    if curr_accuracy>best_accuracy:
                        best_accuracy = curr_accuracy
                        parameter = [Lr,Hidden_size,Reg]
                        save_pic_data(train_losses,validate_losses,accuracies)
                        model.save("best_model.npz")

        print(f"lr: {lr},batch_size: {batch_size},reg: {reg} is the best parameter")
    else:
        model = NeuralNetwork(input_size,hidden_size,output_size)
        if load:
            model.load("best_model.npz")
        else:
            curr_accuracy,train_losses,validate_losses,accuracies = train(model,train_data,val_data,epochs,lr,lr_decay,reg)
            save_pic_data(train_losses,validate_losses,accuracies)
        test_loss,test_accuracy = test(model,X_test,y_test)
        print(f"The accuracy of test_data: {test_accuracy*100}%")

    if visual:
        pic_data = pd.read_csv(f"models/best_model.csv")
        visualize(model, pic_data)

if __name__ == '__main__':
    main()
