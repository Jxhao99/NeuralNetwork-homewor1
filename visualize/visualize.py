import matplotlib.pyplot as plt
import pandas as pd

def visualize(model, log):
    # 可视化训练和测试的loss曲线
    log[['train_loss','validate_loss']].plot(title='train/validate loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("visualize/loss.png")
    plt.close()

    # 可视化测试的accuracy曲线
    log[['validate_accuracy']].plot(title='validate accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("visualize/accuracy.png")
    plt.close()

    # 可视化每层的网络参数
    layer1_weights = model.weights["w1"].flatten().tolist()
    plt.hist(layer1_weights, bins=100)
    plt.title("layer1 weights")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("visualize/layer1_weights.png")
    plt.close()

    layer2_weights = model.weights["w2"].flatten().tolist()
    plt.hist(layer2_weights, bins=30)
    plt.title("layer2 weights")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("visualize/layer2_weights.png")
    plt.close()

    layer1_biases = model.biases["b1"].flatten().tolist()
    plt.hist(layer1_biases, bins=10)
    plt.title("layer1 biases")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("visualize/layer1_biases.png")
    plt.close()

    layer2_biases = model.biases["b2"].flatten().tolist()
    plt.hist(layer2_biases, bins=10)
    plt.title("layer2 biases")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("visualize/layer2_biases.png")
    plt.close()


def save_pic_data(train_losses,validate_losses,accuracies):
    data = {
        "train_loss": train_losses,
        "validate_loss": validate_losses,
        "validate_accuracy": accuracies
    }
    pd.DataFrame(data).to_csv('models/best_model.csv',)
