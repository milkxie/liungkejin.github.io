---
layout: post_layout
title: UFDLD Tutorial Softmax Regression
time: 2016.8.4 Thursday
location: Nanjing
pulished: true
excerpt_separator: "##"
---

## Hypothesis

In the softmax regression setting, we are interested in multi-class classification, and so the **label** $\color{black}{y}$ can take on $\color{black}{K}$ different values, rather than only two. Thus, in our **training set** $\color{black}{\{(x^{(i)},y^{(1)}),...,(x^{(m)},y^{(m)})\},}$ the corresponding $\color{black}{y^{(i)}\in\{1,2,...,K\}}.$

Given a test input $\color{black}{x}$, we need a hypothesis to estimate the probability that 
$\color{black}{P(y=k|x)}$ for each value of $\color{black}{k=1,...,K.}$ In other words,we want to estimate the probability of the class label taking on each of the $\color{black}{K}$ different possible values. So the ouput of our hypothesis must be a $\color{black}{K}$-dimentional vector, like this

$$\color{black}{h(\theta)=\begin{bmatrix}P(y=1|x;\theta)\\
                           P(y=2|x;\theta)\\
                           \vdots         \\
                           P(y=K|x;\theta)\end{bmatrix}
           =\dfrac{1}{\sum_{j=1}^{K}{\rm{exp}}(\theta^{(j){\rm{T}}}x)}
            \begin{bmatrix}{\rm{exp}}(\theta^{(1){\rm{T}}}x)\\
                           {\rm{exp}}(\theta^{(2){\rm{T}}}x)\\
                           \vdots         \\
                           {\rm{exp}}(\theta^{(K){\rm{T}}}x)\end{bmatrix}}$$

Here $\color{black}{\theta^{(1)},\theta^{(2)},...,\theta^{(K)}\in \mathbb{R}}$ are the parameters which require by learning. Notice that the term  $\color{black}{\dfrac{1}{\sum_{j=1}^{K}{\rm{exp}}(\theta^{(j){\rm{T}}}x)}}$ normalizes the distribution, so that it sums to one.


## Cost Function

We now define the **cost function** that we'll use for soft regression

$$\color{black}{J(\theta)=-\Bigg{[}\sum\limits_{i=1}^m\sum\limits_{k=1}^K1\bigg\{y^{(i)}=k\bigg\}{\rm{log}}\frac{\exp(\theta^{(k){\rm{T}}}x^{(i)})}{\sum_{j=1}^K\exp(\theta^{(j){\rm{T}}}x^{(i)})}\Bigg{]}}
$$

Note that the **logistic regression** is a special case of this

$$\color{black}{\begin{align}
J(\theta)
&=-\Bigg{[}\sum\limits_{i=1}^m(1-y^{(i)}){\rm{log}}(1-h_\theta(x^{(i)})+y^{(i)}{\rm{log}}h_\theta(x^{(i)})\Bigg{]}\\
&=-\Bigg{[}\sum\limits_{i=1}^{m}\sum\limits_{k=0}^11\bigg\{y^{(i)}=k\bigg\}{\rm{log}}P(y^{(i)}=k|x^{(i)};\theta)\Bigg{]}\end{align}}
$$


The softmax cost function is similar, except that we now sum over the $\color{black}K$ differente possiblility of the class label. Note that in softmax regression, we have that

$$
\color{black}{P(y^{(i)}=k|x^{(i)};\theta)=\frac{\exp(\theta^{(k){\rm{T}}}x^{(i)})}{\sum_{j=1}^K\exp(\theta^{(j){\rm{T}}}x^{(i)})}}
$$

We can solve the minimum of $\color{black}{J(\theta)}$ iteratively similar to the logistic regression. The gradient is:

$$
\color{black}{\begin{align}
\nabla_{\theta^{(k)}}J(\theta)
&=-\Bigg{[}\sum\limits_{i=1}^m\sum\limits_{k=1}^K1\bigg\{y^{(i)}=k\bigg\}\frac{\sum_{j=1}^K\exp(\theta^{(j){\rm{T}}}x^{(i)})}{\exp(\theta^{(k){\rm{T}}}x^{(i)})}\cdot\frac{\exp(\theta^{(k){\rm{T}}}x^{(i)})x^{(i)}\sum_{j=1}^K\rm(\theta^{(j){\rm{T}}}x^{(i)})-\exp(\theta^{(k){\rm{T}}}x^{(i)})\exp(\theta^{(k){\rm{T}}}x^{(i)})(x^{(i)})}{\bigg{(}\sum_{j=1}^K\exp(\theta^{(j){\rm{T}}}x^{(i)})\bigg{)}^2}\Bigg{]}\\
&=-\sum\limits_{i=1}^mx^{(i)}\Bigg{[}\sum\limits_{k=1}^K1\bigg\{y^{(i)}=k\bigg\}-\sum\limits_{k=1}^K\bigg\{y^{(i)}=k\bigg\}P(y^{(i)}=k|x^{(i)};\theta)\Bigg{]}\\
&=-\sum\limits_{i=1}^mx^{(i)}\Bigg{[}\bigg\{y^{(i)}=k\bigg\}-P(y^{(i)}=k|x^{(i)};\theta)
\Bigg{]}\end{align}}
$$

## Properties of softmax regression parameterization

Softmax regression has a unusual property that we can use it to simplify our parameters $\color{black}{\theta.}$ By subtracting some fixed vector $\color{black}{\psi,}$ we obtain the form $\color{black}{(\theta^{(j)}-\psi)}$ for every $\color{black}{\theta^{(j)}.}$ Our hypothesis now estimates the specific class probability as

$$
\color{black}{\begin{align}
P(y^{(i)}=k\mid x^{(i)};\theta)&=\frac
{\exp((\theta^{(k)}-\psi)^{\rm{T}}x^{(i)})}
{\sum_{j=1}^K\exp((\theta^{(j)}-\psi)^{(T)}x^{(i)})}\\
&=\frac
{\exp(\theta^{(k){\rm{T}}}x^{(i)})\exp(-\psi^{\rm{T}}x^{(i)})}
{\sum_{j=1}^K\exp(\theta^{(j)\rm{T}}x^{(i)})\exp(-\psi^{\rm{T}}x^{(i)})}\\
&=\frac
{\exp(\theta^{(k)\rm{T}}x^{(i)})}{\sum_{j=1}^K\exp(\theta^{(j)\rm{T}}x^{(i)})}
\end{align}}
$$   

The term "overparameterized" is used to describe the situation in which substracting
$\color{black}{\psi}$ dose not affect our hypothesis predictions at all! There are multiole parameter settings can resulting in the same hypothesis function 
$\color{black}{h_\theta}$.

In the case, the minimizer of the $\color{black}{J(\theta)}$ is not unique. We can also minimize $\color{black}{J(\theta)}$ by 
$\color{black}{(\theta^{(1)}-\psi,\theta^{(2)}-\psi,...,\theta^{(k)}-\psi)}$ for any value of $\color{black}{\psi}$ to get the same result.(Interestingly, 
$\color{black}{J(\theta)}$ is still convex, and thus gradient descent will not run into local optima problems. But the Hessian is singular/non-invertible, which causes a straightforward implementation of Newton’s method to run into numerical problems.)

## Relationship to Logistic Regression

It is easy to show that the **softmax regression** is a generalization of **logstic regression**. When $\color{black}{K=2}$, we assum the setting
$\color{black}{\psi=\theta^{(2)}}$ 

$$\color{black}
{\begin{align}
h_\theta(x)
&=\frac
{1}{\exp(\theta^{(1)\rm{T}}x^{(i)})+\exp(\theta^{(2)\rm{T}}x^{(i)})}\cdot
\begin{bmatrix}
\exp(\theta^{(1)\rm{T}}x^{(i)})\\
\exp(\theta^{(2)\rm{T}}x^{(i)})
\end{bmatrix}\\
&=\frac
{1}{\exp((\theta^{(1)\rm{T}}-\theta^{(2)\rm{T}})x^{(i)})+\exp(\vec{0}\cdot x^{(i)})}\cdot
\begin{bmatrix}
\exp((\theta^{(1)\rm{T}}-\theta^{(2)\rm{T}})x^{(i)})\\
\exp(\vec{0}\cdot x^{(i)})
\end{bmatrix}\\
&=
\begin{bmatrix}
\frac
{1}{1+\exp((\theta^{(1)\rm{T}}-\theta^{(2)\rm{T}})x^{(i)})}\\
1-\frac
{1}{1+\exp((\theta^{(1)\rm{T}}-\theta^{(2)\rm{T}})x^{(i)})}
\end{bmatrix}
\end{align}}
$$

Furthermore, replacing $\color{black}{\theta^{(1)}-\theta^{(2)}}$ with a single parameter vector $\color{black}{-\theta',}$ we find that the **softmax regression** reduces to the **logistic regression**.





















                           



