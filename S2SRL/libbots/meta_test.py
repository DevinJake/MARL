import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.optim as optim
from torch.autograd import Variable

"""
This solution seems to work so far. 
The idea is to first remove all Parameters in the Module, and replace them by tensors under object attributes with the same name. 
Here is what I used:
"""
def flip_parameters_to_tensors(module):
    attr = []
    while bool(module._parameters):
        attr.append( module._parameters.popitem() )
    setattr(module, 'registered_parameters_name', [])

    for i in attr:
        setattr(module, i[0], torch.zeros(i[1].shape,requires_grad=True))
        module.registered_parameters_name.append(i[0])

    module_name = [k for k,v in module._modules.items()]

    for name in module_name:
        flip_parameters_to_tensors(module._modules[name])

"""
Then, we can used the saved list of previously active attributes to assign the tensors.
This way, the flattened vector is assigned to all tensors of the NN for evaluation. 
The backward() gets back to the parameter vector outside the NN.
"""
def set_all_parameters(module, theta):
    count = 0

    for name in module.registered_parameters_name:
        a = count
        b = a + getattr(module, name).numel()
        t = torch.reshape(theta[0,a:b], getattr(module, name).shape)
        setattr(module, name, t)

        count += getattr(module, name).numel()

    module_name = [k for k,v in module._modules.items()]
    for name in module_name:
        count += set_all_parameters(module._modules[name], theta)
    return count

def gradient_test():
    # 第一个卷积层，我们可以看到它的权值是随机初始化的；
    w = nn.Linear(1, 1)
    x = torch.tensor([1.0], requires_grad=False)
    y = torch.tensor(2.0, requires_grad=False)
    print('w.weight: ')
    print(w.weight)
    print('w.bias: ')
    print(w.bias)
    weight0 = w.weight
    loss = (y - w(x)) * (y - w(x))
    print('loss: ')
    print(loss)
    w.zero_grad()
    grads = torch.autograd.grad(loss, w.weight, create_graph=True)
    print('grads:')
    print(grads)
    new_weight = w.weight - grads[0]
    print('w.weight:')
    print(w.weight)
    print('new_weight:')
    print(new_weight)
    print('w.new_bias: ')
    print(w.bias)
    print('###########################################')

    # note: can assign parameters to nn.Module and track the original grad_fn of the parameters
    del w.weight
    w.weight = new_weight
    weight1 = new_weight
    print('w.weight: ')
    print(w.weight)
    x = torch.tensor([2.0], requires_grad=False)
    y = torch.tensor(3.0, requires_grad=False)
    loss = (y - w(x)) * (y - w(x))
    print('loss: ')
    print(loss)
    w.zero_grad()
    grads = torch.autograd.grad(loss, w.weight, create_graph=True)
    new_weight = w.weight - grads[0]
    print('grads:')
    print(grads)
    maml_grads = torch.autograd.grad(loss, weight0, create_graph=True)
    print('maml_grads:')
    print(maml_grads)
    print('w.weight:')
    print(w.weight)
    print('new_weight:')
    print(new_weight)
    print('w.new_bias: ')
    print(w.bias)
    print('###########################################')

    del w.weight
    w.weight = new_weight
    print('w.weight: ')
    print(w.weight)
    x = torch.tensor([3.0], requires_grad=False)
    y = torch.tensor(4.0, requires_grad=False)
    loss = (y - w(x)) * (y - w(x))
    print('loss: ')
    print(loss)
    w.zero_grad()
    new_weight = w.weight - grads[0]
    print('grads:')
    print(grads)
    w1_grads = torch.autograd.grad(loss, weight1, create_graph=True)
    print('weight1_grads:')
    print(w1_grads)
    maml_grads = torch.autograd.grad(loss, weight0, create_graph=True)
    print('maml_grads:')
    print(maml_grads)
    print('w.weight:')
    print(w.weight)
    print('new_weight:')
    print(new_weight)
    print('w.new_bias: ')
    print(w.bias)
    print('###########################################')

def detach_test():
    x = torch.tensor(([2.0]), requires_grad=True)
    xx = torch.tensor(([2.0]), requires_grad=True)
    yy = torch.tensor(([1.0]), requires_grad=True)
    x.data = xx.clone().detach() - yy.clone().detach()
    y = x ** 2
    z = 2 * y
    w = z ** 3

    # This is the subpath
    # Do not use detach()
    p = z
    # p = z.detach()
    q = torch.tensor(([2.0]), requires_grad=True)
    pq = p * q
    pq.backward(retain_graph=True)

    w.backward()
    print('x.grad:')
    print(x.grad)
    print('xx.grad:')
    print(xx.grad)
    print('yy.grad:')
    print(yy.grad)
    # x.data = x.data-x.grad.data
    x.data -= x.grad.data
    print('x:')
    print(x)
    print('xx:')
    print(xx)

if __name__ == "__main__":

    # gradient_test()
    input_shapes = ((1, 3, 3, 4),)
    for s in input_shapes:
        print(s)
    a = tuple(torch.randn(s) for s in input_shapes)
    print(a)
    print(*a)

    a = list()
    temp = torch.tensor(2.0, requires_grad=True)
    temp = temp.clone().detach()
    a.append(temp)
    print(np.mean(a))

    tensor_list = []
    t1 = torch.tensor([[1.0,2.0],[3.0,4.0]], requires_grad=True)
    tensor_list.append(t1)
    t2 = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], requires_grad=True)
    tensor_list.append(t2)
    t3 = torch.tensor([[-1.0, -1], [-1, -1]], requires_grad=True)
    tensor_list.append(t3)
    bb = torch.stack(tensor_list).mean(0)
    aa = bb.detach()
    print(tensor_list)
    print(aa)
    cc = t1.detach() - aa
    print(cc)

    detach_test()




