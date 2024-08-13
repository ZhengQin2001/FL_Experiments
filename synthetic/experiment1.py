import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import get_probabilities_extended, get_bs_optimal

def main():
    # Define parameters
    param_pyxa = []
    param_pyxa.append([-0.25,0,0.25])
    param_pyxa.append([0.1,0.1,0.1])
    param_pyxa.append([0.9,0.9,0.9])
    param_pyxa = np.array(param_pyxa).transpose()

    th = np.array([0])
    param_pyxa_dic = {}
    param = []
    param.append(th-0.25)
    param.append(np.array([0.9,0.1]))
    param_pyxa_dic[0] = param
    param = []
    param.append(th)
    param.append(np.array([0.9,0.1]))
    param_pyxa_dic[1] = param
    param = []
    param.append(th+0.25)
    param.append(np.array([0.8,0.2]))
    param_pyxa_dic[2] = param

    param_pxa = []
    param_pxa.append([-0.5,0,0.5])
    param_pxa.append([1,1,1])
    param_pxa = np.array(param_pxa).transpose()

    # plot

    param_pa = np.array([1,1,1])/3

    x_array = np.linspace(-4,4,2000)
    dx = np.max(x_array)-np.min(x_array)
    dx /= x_array.shape[0]
    p_xa, p_yxa = get_probabilities_extended(x_array, param_pa, param_pyxa_dic, param_pxa)

    mua = np.array([1,1,1])
    mua = mua/np.sum(mua)
    type_str = 'MSE'
    h,risk,_ = get_bs_optimal(x_array,p_xa,p_yxa,mua,type=type_str)

    print('mu_a : ',mua)
    print('Risks : ', risk)

    plt.figure(figsize=(10,2))
    for i in np.arange(param_pa.shape[0]):
        plt.plot(x_array,p_yxa[:,i],label = 'p(y|x,'+str(i)+')')
    plt.plot(x_array,h,label = 'balanced h')
    plt.xlabel('x')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,2))
    for i in np.arange(param_pa.shape[0]):
        plt.plot(x_array,p_xa[:,i],label = 'p(x|'+str(i)+')')
    plt.xlabel('x')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
