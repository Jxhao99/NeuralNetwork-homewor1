import numpy as np
import os
class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.weights ={
            "w1":np.random.randn(input_size, hidden_size)/np.sqrt(input_size),
            "w2":np.random.randn(hidden_size, output_size)/np.sqrt(hidden_size)
        }
        self.biases ={
            "b1":np.random.randn(hidden_size),
            "b2":np.random.randn(output_size)
        }

        self.linear_transforms = [np.zeros(hidden_size) for _ in range(2)]
        self.activations =  [np.zeros(output_size) for _ in range(3)]

    def forward(self, input):
        '''
        前向传播
        '''
        self.activations[0] = input
        w1,w2,b1,b2 = self.weights["w1"],self.weights["w2"],self.biases["b1"],self.biases["b2"]
        self.linear_transforms[0] = np.dot(input,w1)+b1
        self.activations[1] = np.maximum(0, self.linear_transforms[0])
        self.linear_transforms[1] = np.dot(self.activations[1],w2)+b2
        exp_scores = np.exp(self.linear_transforms[1])
        self.activations[2] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.activations[-1]

    def backward(self,out_put,y):
        grads = {}
        N = out_put.shape[0]
        dscore = out_put                                 # (N,C)
        dscore[range(N), y] -= 1

        grads['w2'] = np.dot(self.activations[1].T, dscore)/N
        grads['b2'] = np.sum(dscore, axis = 0)/N

        dhidden = np.dot(dscore, self.weights['w2'].T)
        dhidden[self.activations[1] <=0] = 0

        grads['w1'] = np.dot(self.activations[0].T, dhidden)/N
        grads['b1'] = np.sum(dhidden, axis = 0)/N

        return grads

    def CrossEntropyLoss(self,X_pre, y=None,reg=1):
        W1, b1 = self.weights['w1'], self.biases['b1']
        W2, b2 = self.weights['w2'], self.biases['b2']
        N = X_pre.shape[0]

        probs = X_pre / np.sum(X_pre, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * reg * np.sum(W1*W1) + 0.5 * reg * np.sum(W2*W2)
        loss = data_loss + reg_loss
        return data_loss

    def SGD(self,grads,learning_rate,reg):
        lr_reg = 1-learning_rate*reg
        self.weights['w1'] =-learning_rate*(grads['w1'])+lr_reg*self.weights['w1']
        self.biases['b1'] = -learning_rate*(grads['b1'])+lr_reg*self.biases['b1']
        self.weights['w2'] =-learning_rate*(grads['w2'])+lr_reg*self.weights['w2']
        self.biases['b2'] = -learning_rate*(grads['b2'])+lr_reg*self.biases['b2']

    def save(self, filename):
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=[self.weights[_] for _ in self.weights],
            biases=[self.biases[_] for _ in self.biases],
        )

    def load(self, filename):
        npz_members = np.load(os.path.join(os.curdir, 'models', filename), allow_pickle=True)

        self.weights["w1"] = list(npz_members['weights'])[0]
        self.weights["w2"] = list(npz_members['weights'])[1]
        self.biases["b1"]  = list(npz_members['biases'])[0]
        self.biases["b2"]  = list(npz_members['biases'])[1]
