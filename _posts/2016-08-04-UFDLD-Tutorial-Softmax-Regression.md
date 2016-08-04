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


