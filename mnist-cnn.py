import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(x, 0)

def dxrelu(x):
    return (x > 0).astype(float)

def layer_1_fp(X,W1,B1):
    X1 = np.dot(W1,X) + B1
    A2 = relu(X1)
    return X1,A2


def layer_2_fp(A2,W2,B2):
    X2 = np.dot(W2,A2) + B2
    A3 = relu(X2)
    return X2,A3

def cost(A3,label):
    y_h = np.zeros((10))
    y_h[label] = 1
    c = A3 - y_h
    c = np.square(c)
    cost = np.sum(c) 
    return cost



def back_prop(W1,B1,W2,B2,A1,A2,A3,X1,X2,alpha,label):
    
    y_h = np.zeros((10))
    y_h[label ] = 1
    error = A3 - y_h
    dW2 = np.dot((2 * error * dxrelu(X2)).reshape(10,1),A2.reshape(10,1).transpose()) #10 x 10
    dB2 = 2 * error * dxrelu(X2) # 10, 1


    temp1  =  (np.dot(dB2 ,W2) * dxrelu(X1)).reshape(10,1)    
    temp2 =   A1.reshape((784,1)).transpose()
    dW1 = np.dot(temp1,temp2) # 10 *784
    dB1 = np.dot(dB2 ,W2) * dxrelu(X1)

    W1 += -alpha * dW1
    B1 += -alpha * dB1
    W2 += -alpha * dW2
    B2 += -alpha * dB2
    
    return W1, B1, W2, B2


def save_weights_and_biases(W1, B1, W2, B2, filename):
    """
    Save weights and biases to a text file.

    Parameters:
    - W1 (numpy.ndarray): Weights of the first layer.
    - B1 (numpy.ndarray): Biases of the first layer.
    - W2 (numpy.ndarray): Weights of the second layer.
    - B2 (numpy.ndarray): Biases of the second layer.
    - filename (str): The file where the data will be saved.
    """
    with open(filename, 'w') as file:
        file.write('Weights and Biases:\n\n')

        file.write('Weights W1:\n')
        np.savetxt(file, W1, fmt='%.6f', header='Shape: {}'.format(W1.shape))

        file.write('\nBiases B1:\n')
        np.savetxt(file, B1, fmt='%.6f', header='Shape: {}'.format(B1.shape))

        file.write('\nWeights W2:\n')
        np.savetxt(file, W2, fmt='%.6f', header='Shape: {}'.format(W2.shape))

        file.write('\nBiases B2:\n')
        np.savetxt(file, B2, fmt='%.6f', header='Shape: {}'.format(B2.shape))



def training():
    epochs = 750
    alpha = 0.00001
    costls = []
    W1 = np.random.uniform(-0.02, 0.02, (10, 784))
    B1 = np.random.uniform(-0.02, 0.02, (10))
    
    W2 = np.random.uniform(-0.02, 0.02, (10, 10))
    B2 = np.random.uniform(-0.02, 0.02, (10))
    prev_cost = 0
    cost_rpt = 0
    for i in range(epochs):
     
        
        train =r"/content/train.csv"

        print("EPOCH ",i,":")

        dataset = pd.read_csv(train)
        current_cost = 0
        

        n = len(dataset)
        for i in range(n):
            label = dataset.iloc[i,0]
            pxl = np.array(dataset.iloc[i,1:])


            X1,A2 = layer_1_fp(pxl,W1,B1)
            X2,A3 = layer_2_fp(A2,W2,B2)
            current_cost += cost(A3,label)
            
            
            W1, B1, W2, B2 = back_prop(W1,B1,W2,B2,pxl,A2,A3,X1,X2,alpha,label)

        save_weights_and_biases(W1, B1, W2, B2, 'Weights.txt')        
        current_cost /= n
        print("Cost: ",current_cost)

        if current_cost == prev_cost:
            cost_rpt += 1
        else:
            cost_rpt = 0
        if cost_rpt == 5:
            print("NO CHANGE IN COST")
            break
        costls.append(current_cost)
        print("Cost difference: ",current_cost - prev_cost)
        print("\n")
        prev_cost = current_cost
    import matplotlib.pyplot as plt
    
    plt.plot(costls)
    plt.show()

    return W1,B1,W2,B2
        
W1,B1,W2,B2 = training()

print("Weights saved in Weights.txt")