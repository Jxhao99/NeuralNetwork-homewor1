import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import NeuralNetwork


def train(model,training_data, validation_data, epochs,lr,lr_decay=1,reg=1):
    best_accuracy = 0
    train_loss = 0
    train_losses = []
    validate_losses = []
    accuracies = []
    X_train,y_train = training_data[0],training_data[1]
    X_val,y_val = validation_data[0],validation_data[1]

    for epoch in range(epochs+1):
        for input,label in zip(X_train,y_train):
            output = model.forward(input)
            train_loss += model.CrossEntropyLoss(output,label,reg)
            if epoch>0:
                gradient = model.backward(output,label)
                model.SGD(gradient,lr,reg)
                lr*=lr_decay

        train_loss/=len(X_train)
        validate_loss,validate_accuracy = test(model,X_val,y_val)

        train_losses.append(train_loss)
        validate_losses.append(validate_loss)
        accuracies.append(validate_accuracy)

        if validate_accuracy>best_accuracy:
            best_accuracy = validate_accuracy
            model.save("best_model.npz")
        print(f"****Epoch {epoch}, loss:{validate_loss},accuracy:{validate_accuracy*100} %.")

    return best_accuracy,train_losses,validate_losses,accuracies

def test(model,X_t,y_t):
    scores = model.forward(X_t)
    y_pred = np.argmax(scores, axis = 1)
    test_loss = model.CrossEntropyLoss(scores,y_t)
    test_accuracy = (y_pred == y_t).mean()
    return test_loss,test_accuracy


def save_pic_data(train_losses,validate_losses,accuracies):
    data = {
        "train_loss": train_losses,
        "validate_loss": validate_losses,
        "validate_accuracy": accuracies
    }
    pd.DataFrame(data).to_csv('models/best_model.csv',)
