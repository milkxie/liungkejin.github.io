---
layout: post_layout
title: UFDLD-Tutorial-Softmax-Regression
time: 2016年08月04日 星期四
location: 南京
pulished: true
excerpt_separator: "```"
---

# Softmax Regression

## Hypothesis

In the softmax regression setting, we are interested in multi-class classification, and so the **label** $y$ can take on $K$ different values, rather than only two. Thus, in our **training set** $\{(x^{(i)},y^{(1)}),...,(x^{(m)},y^{(m)})\},$ the corresponding $y^{(i)}\in\{1,2,...,K\}.$

Given a test input $x$, we need a hypothesis to estimate the probability that 
$P(y=k|x)$ for each value of $k=1,...,K.$ In other words,we want to estimate the probability of the class label taking on each of the $K$ different possible values. So the ouput of our hypothesis must be a $K$-dimentional vector, like this

$$h(\theta)=\begin{bmatrix}P(y=1|x;\theta)\\
                           P(y=2|x;\theta)\\
                           \vdots         \\
                           P(y=K|x;\theta)\end{bmatrix}
           =\dfrac{1}{\sum_{j=1}^{K}{\rm{exp}}(\theta^{(j){\rm{T}}}x)}
            \begin{bmatrix}{\rm{exp}}(\theta^{(1){\rm{T}}}x)\\
                           {\rm{exp}}(\theta^{(2){\rm{T}}}x)\\
                           \vdots         \\
                           {\rm{exp}}(\theta^{(K){\rm{T}}}x)\end{bmatrix}$$

Here $\theta^{(1)},\theta^{(2)},...,\theta^{(K)}\in \rm I\!R$ are the parameters which require by learning. Notice that the term  $\dfrac{1}{\sum_{j=1}^{K}{\rm{exp}}(\theta^{(j){\rm{T}}}x)}$ normalizes the distribution, so that it sums to one.

## Cost Function

We now define the **cost function** that we'll use for soft regression

$$J(\theta)=-\Bigg{[}\sum\limits_{i=1}^m\sum\limits_{k=1}^K1\bigg\{y^{(i)}=k\bigg\}{\rm{log}}\dfrac{{\rm{exp}}(\theta^{(k){\rm{T}}}x^{(i)})}{\sum_{j=1}^K{\rm{exp}}(\theta^{(j){\rm{T}}}x^{(i)})}\Bigg{]}
$$

Note that the **logistic regression** is a special case of this

$$\begin{align}
J(\theta)
&=-\Bigg{[}\sum\limits_{i=1}^m(1-y^{(i)}){\rm{log}}(1-h_\theta(x^{(i)})+y^{(i)}{\rm{log}}h_\theta(x^{(i)})\Bigg{]}\\
&=-\Bigg{[}\sum\limits_{i=1}^{m}\sum\limits_{k=0}^11\bigg\{y^{(i)}=k\bigg\}{\rm{log}}P(y^{(i)}=k|x^{(i)};\theta)\Bigg{]}\end{align}
$$


The softmax cost function is similar, except that we now sum over the $K$ differente possiblility of the class label. Note that in softmax regression, we have that

$$P(y^{(i)}=k|x^{(i)};\theta)=\dfrac{{\rm{exp}}(\theta^{(k){\rm{T}}}x^{(i)})}{\sum_{j=1}^K{\rm{exp}}(\theta^{(j){\rm{T}}}x^{(i)})}$$

We can solve the minimum of $J(\theta)$ iteratively similar to the logistic regression. The gradient is:

$$\begin{align}
\nabla_{\theta^{(k)}}J(\theta)
&=-\Bigg{[}\sum\limits_{i=1}^m\sum\limits_{k=1}^K1\bigg\{y^{(i)}=k\bigg\}\dfrac{\sum_{j=1}^K{\rm{exp}}(\theta^{(j)T}x^{(i)})}{{\rm{exp}}(\theta^{(k)T}x^{(i)})}\cdot\dfrac{{\rm{exp}}(\theta^{(k)T}x^{(i)})x^{(i)}\sum_{j=1}^K{\rm{exp}}(\theta^{(j)T}x^{(i)})-{\rm{exp}}(\theta^{(k)T}x^{(i)}){\rm{exp}}(\theta^{(k)T}x^{(i)})(x^{(i)})}{\bigg{(}\sum_{j=1}^K{\rm{exp}}(\theta^{(j)T}x^{(i)})\bigg{)}^2}\Bigg{]}\\
&=-\sum\limits_{i=1}^mx^{(i)}\Bigg{[}\sum\limits_{k=1}^K1\bigg\{y^{(i)}=k\bigg\}-\sum\limits_{k=1}^K\bigg\{y^{(i)}=k\bigg\}P(y^{(i)}=k|x^{(i)};\theta)\Bigg{]}\\
&=-\sum\limits_{i=1}^mx^{(i)}\Bigg{[}\bigg\{y^{(i)}=k\bigg\}-P(y^{(i)}=k|x^{(i)};\theta)\Bigg{]}\end{align}$$





















                           



