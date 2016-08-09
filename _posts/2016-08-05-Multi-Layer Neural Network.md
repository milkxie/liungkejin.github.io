---
layout: post_layout
title: UFDLD Tutorial Multi-Layer Neural Network
time: 2016年08月04日 星期四
location: 南京
published: true
excerpt_separator: "```"
---

# Introduction

There are some complex learning problems that we fail to use **linear regression** to solve perfectly. In this case, the **Neural Network** with a non-linear form of hypothesis $\color{black}{h_{W,b}(x)}$ fits our data via learning the **parameter** 
$\color{black}{W,b.}$

We first describe the simplest neural network, one which comprise a single **neuron**, just like the following diagram.

<img src="<img src="/assets/img/single_neuron.png" width="400px" />" />

This "neuron" is a computational unit that take as input $\color{black}{x_1,x_2,x_3}$(and a +1 intercept term), and outputs $\color{black}{h_{W,b}(x)}$, where

$$
\color{black}{
h_{W,b}(x)=f(W^Tx)=f(\sum_{i=1}^3W_ix_i+b),
}
$$

$\color{black}{f(\cdot)}$ in the above equation is called **activation function**. There are several choice about it

```

**sigmoid function**:

$$\color{black}{
f(z)=\frac{1}{1+\exp(-z)}}
$$

**tanh function** ( also called **hyperbolic tangent** )

$$\color{black}{
f(z)=\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}}
$$

**rectified linear activetion function**

$$\color{black}{
f(z)=\max(0,x)}
$$

Note that we use the sigmoid function almost time here.

Here are plots of the three mentioned functions:

<img src="<img src="/assets/img/activation_functions.png" width="400px" />" />


# Neural Network model

A neural network comprises many of simple neurons, so that the output of a neuron can be the input of another.

<img src="<img src="/assets/img/simple_network.png" width="400px" />" />

The circles labeled "+1" are called **bias units**.The leftmost layer of the network is called the **input layer**, and the rightmost layer the **output layer**. The middle layer of nodes is called the **hidden layer** ( possibly not only one layer), because its values are not observed in the **training set**.   

We will let $\color{black}{n_l}$ denote the number of layers, thus 
$\color{black}{n_l=3}$ in our example. We label layer $\color{black}{l}$ as 
$\color{black}{L_l}$. The neural network has parameters $\color{black}{(W,b),}$ where we write $\color{black}{W_{ij}^{(l)}}$ to denote the parameter (or weight) associated with the connection between unit $\color{black}{j}$ in layer $\color{black}{l,}$ and unit $\color{black}{i}$ in layer $\color{black}{l+1.}$ Also, 
$\color{black}{b_i^{(l)}}$ is the **bias** associated with units $\color{black}{i}$ in the layer $\color{black}{l+1.}$. Thus, in our example, we have 
$\color{black}{W^{(1)}\in \mathbf R^{3\times 3},}$ and 
$\color{black}{W^{(2)}\in \mathbf R^{1\times 3}.}$ We also let $\color{black}{s_l}$ denote the number of nodes in layer $\color{black}{l}$ (not counting the bias unit).

We will write $\color{black}{a_i^{(l)}}$ to denote the **activation** of unit 
$\color{black}{i}$ in layer $\color{black}{l.}$ Conventionally, we also use 
$\color{black}{a_i^{(1)}=x_i.}$ In this example, we show the computations of our neural network with a fixed setting of the parameters $color{black}{W,b}$

$$\color{black}{\begin{align}
a_1^{(2)}&=f(W_{11}^{(1)}x_1+W_{12}^{(1)}x_2+W_{13}^{(1)}x_3+b_1^{(1)})\\
a_2^{(2)}&=f(W_{21}^{(1)}x_1+W_{22}^{(1)}x_2+W_{23}^{(1)}x_3+b_2^{(1)})\\
a_3^{(2)}&=f(W_{31}^{(1)}x_1+W_{32}^{(1)}x_2+W_{33}^{(1)}x_3+b_3^{(1)})\\
h_{W,b}(x)&=a_1^{(3)}=f(W_{11}^{(2)}a_1^{(2)}+W_{12}^{(2)}a_2^{(2)}+W_{13}^{(2)}a_3^{(2)}+b_1^{(2)})
\end{align}}$$

Compactly, we define

$$\color{black}{
z_i^{(l)}=\sum_{j=1}^{(n)}W_{ij}^{(l-1)}a_j^{(l-1)}+b_i^{(l-1)}
}$$

So that

$$\color{black}{
a_i^{(l)}=f(z_i^{(l)}).
}$$

Now we show the **forward propagation** in this example

$$\color{black}{\begin{align}
z^{(2)}&=W^{(1)}x+b^{(1)}\\
a^{(2)}&=f(z^{(2)})\\
z^{(3)}&=W^{(2)}a^{(2)}+b^{(2)}\\
h_{W,b}(x)&=a^{(3)}=f(z^{(3)})
\end{align}}$$


# Backpropagation Algorithm

Given a training set $\color{black}{\{(x^{(1)},y^{(1)}),...,(x^{(m)},y^{(m)})\}}$ of $\color{black}{m}$ training example, we can define the cost function using the error measure **(one-half) squared-error cost**

$$\color{black}{\begin{align}
J(W,b)
&=\Bigg{[}\frac{1}{m}\sum\limits_{i=1}^{m}J(W,b;x^{(i)},y^{(i)})
+\frac{\lambda}{2}\sum\limits_{l=1}^{n_l-1}\sum\limits_{i=1}^{s_l}\sum\limits_{j=1}^{s_l+1}(W_{ji}^{(l)})^2\Bigg{]}\\
&=\Bigg{[}\frac{1}{m}\sum\limits_{i=1}^{m}\Big{(}\frac{1}{2}
\left \lVert y^{(i)}-h_{W,b}(x^{(i)})\right \rVert^2\Big{)}\Bigg{]}
+\frac{\lambda}{2}\sum\limits_{l=1}^{n_l-1}\sum\limits_{i=1}^{s_l}\sum\limits_{j=1}^{s_l+1}(W_{ji}^{(l)})^2
\end{align}}$$

The first term in the definition is an average sum-of-squares error term, while the other is a regularization term (also called a **weight decay** term) that tends to decrease the magnitude of the weights, and helps prevent overfitting. And the **weight decay parameter** $\color{black}{\lambda}$ controls the relative importance of the two terms.

Our goal is to minimize $\color{black}{J(W,b).}$ To train our neural network, we will initialize each parameter $\color{black}{W_{ij}^{l}}$ and $\color{black}{b_i^{(l)}}$ to a small random value near zero (say according to a $\color{black}{Normal(0,\epsilon^2)}$ distribution for some small $\color{black}{\epsilon,}$ say 0.01), and then apply an optimization algorithm such as batch gradient descent. Since 
$\color{black}{J(W,b)}$ is a non-convex function, gradient descent is susceptible to local optima; however, in practice gradient descent usually works fairly well. Finally, note that it is important to initialize the parameters randomly, rather than to all 0’s. If all the parameters start off at identical values, then all the hidden layer units will end up learning the same function of the input.

One iteration of gradient descent updates the parameters $\color{black}{W,b}$ as follow:

$$\color{black}{\begin{align}
W_{ij}^{(l)}&=W_{ij}^{(l)}-\alpha\frac{\partial}{\partial W_{ij}^{(l)}}J(W,b)\\
b_i^{(l)}&=b_i^{(l)}-\alpha\frac{\partial}{\partial b_i^{(l)}}J(W,b)
\end{align}}$$

where $\color{black}{\alpha}$ is the **learning rate**. The key step is computing the partial derivatives above, which is solved efficiently by the **backpropagation** algorithm.

We now describe the algorithm step by step. Obviously, the derivative of the overall cost function $\color{black}{J(W,b)}$ can be computed as:

$$\color{black}{\begin{align}
\frac{\partial}{\partial W_{ij}^{(l)}}J(W,b)
&=\Bigg{[}\frac{1}{m}\sum\limits_{i=1}^m\frac{\partial}{\partial W_{ij}^{(l)}}J(W,b;x^{(i)},y^{(i)})\Bigg{]}+\lambda W_{ij}^{(l)}\\
\frac{\partial}{\partial b_{i}^{(l)}}J(W,b)
&=\frac{1}{m}\sum\limits_{i=1}^m\frac{\partial}{\partial b_{i}^{(l)}}J(W,b;x^{(i)},y^{(i)})
\end{align}}$$

The two lines above differ slightly because weight decay is applied to 
$\color{black}{W}$ but not $\color{black}{b.}$

The intuition behind the backpropagation algorithm is as follows. Given a training example $\color{black}{(x,y),}$ we will first run a **"forward pass"** to compute all the activations throughout the network, including the output value of the hypothesis 
$\color{black}{h_{W,b}(x).}$ Then, for each node $\color{black}{i}$ in layer 
$\color{black}{l,}$ we would like to compute an **"error term"** 
$\color{black}{\delta_i^{(l)}}$ that measures favor that node was "responsible" for any errors in our output. In detail, here is the **backpropagation algorithm**:


1. Perform a feedforward pass, computing the activations for layers 
$\color{black}{L_2,L_3,}$ and so on up to the output layer $\color{black}{L_{n_l}.}$

2. For each output unit $\color{black}{i}$ in layer $\color{black}{n_l}$ (the output layer), set
$$\color{black}{
\delta_i^{(n_l)}=\frac{\partial}{\partial z_i^{(n_l)}}\frac{1}{2}\left \lVert y-h_{W,b}(x)\right \rVert^2=-(y_i-a_i^{(n_l)})\cdot f'(z_i^{(n_l)})
}$$  

3. For $\color{black}{l=n_l-1,n_l-2,n_l-3,...,2}$     
   &nbsp;&nbsp;&nbsp;&nbsp;For each node $\color{black}{i}$ in layer 
   $\color{black}{l}$ 
   
   $$\color{black}{
   \delta_i^{(l)}=(\sum_{j=1}^{s_{l+1}}W_{ji}^{(l)}\delta_j^{(l+1)})\cdot f'(z_i^{(l)})
   }$$
   
4. Compute the desired partial derivatives, which are given as:

$$\color{black}{\begin{align}
\frac{\partial}{\partial W_{ij}^{(l)}}J(W,b;x,y)&=a_j^{(l)}\delta_i^{(l+1)}\\
\frac{\partial}{\partial b_i^{(l)}}J(W,b;x,y)&=\delta_i^{(l+1)}
\end{align}}$$

Intuitively, for a specific parameter $\color{black}{W_{ji}^{(l)}}$, the 3rd step seems to be a iteration process, one which reverse the **"forward pass"** from the output unit back to the activation unit $\color{black}{a_i^{(l)}.}$

Then, we will provide a more mathematic perspective of the **backpropagation** algorithm, one which applies the **chain rule** to compute partial derivatives. For simplicity, we describe the process with a specific example which computes the 
$\color{black}{\frac{\partial}{\partial W_{22}^{(2)}}J(W,b)}$ in the following disagram.

<img src="<img src="/assets/img/multi-neural_network.png" width="400px" />" />

According to statements in the above, we can easily write 
$\color{black}{\frac{\partial}{\partial W_{22}^{(2)}}J(W,b)}$

$$\color{black}{\begin{align}
\frac{\partial}{\partial W_{22}^{(2)}}J(W,b)
&=\Bigg{[}\frac{1}{m}\sum\limits_{i=1}^m\frac{\partial}{\partial W_{22}^{(2)}}J(W,b;x^{(i)},y^{(i)})\Bigg{]}+\lambda W_{22}^{(2)}
\end{align}}$$

where

$$\color{black}{\begin{align}
J(W,b;x^{(i)},y^{(i)})
&=\frac{1}{2}
\left \lVert y^{(i)}-h_{W,b}(x^{(i)})\right \rVert^2\\
h_{W,b}(x^{(i)})
&=a_1^{(4)}=f(z^{(4)})\\
z^{(4)}&=\sum_{j=1}^{s_3}W_{1j}^{(3)}a_j^{(3)}+b_1^{(3)}
\end{align}}$$

When **"forward pass"**, the single parameter $\color{black}{W_{22}^{(2)}}$ respect to the activation $\color{black}{a_2^{(3)}}$, so

$$\color{black}{\begin{align}
a_2^{(3)}
&=f(z^{(3)})\\
z^{(3)}
&=\sum_{j=1}^{s_2}W_{2j}^{(2)}a_j^{(2)}+b_2^{(2)}
\end{align}}$$

Now we apply the **chain rule** to compute the partial derivatives

$$\color{black}{\begin{align}
\frac{\partial}{\partial W_{22}^{(2)}}J(W,b;x^{(i)},y^{(i)})
&=\frac{\partial J}{\partial a_1^{(4)}}\cdot\frac{\partial a_1^{(4)}}{\partial z^{(4)}}
\cdot\frac{\partial z^{(4)}}{\partial a_2^{(3)}}
\cdot\frac{\partial a_2^{(3)}}{\partial z^{(3)}}
\cdot\frac{\partial z^{(3)}}{\partial W_{22}^{(2)}}\\
&=-(y^{i}-a_1^{(4)})\cdot f'(z^{(4)})\cdot W_{12}^{(3)}\cdot f'(z^{(3)})\cdot a_2^{(2)}
\end{align}}$$

where $\color{black}{f'(\cdot)=f(\cdot)\cdot(1-f(\cdot)),}$ because in our example 
$\color{black}{f(\cdot)}$ denote the **sigmoid function**. We also apply the above **backpropagation** algorithm to compute and obtain the same result, which proves that we correctly use the **chain rule** in our specific example. Further, it is easy to extend to every parameter in same way.
