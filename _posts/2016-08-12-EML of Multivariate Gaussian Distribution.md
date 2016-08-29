---
layout: post_layout
title: EML of Multivariate Gaussian Distribution
time: 2016.8.12 Friday
location: Nanjing
pulished: true
excerpt_separator: "##"
---


## Basic Form of Multivariate Gaussian

$$\color{black}{
p(x)=\dfrac{1}{(2\pi)^{D/2}\mid\Sigma\mid^{1/2}}
\exp\Big\{-\dfrac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\Big\}
}$$

$\color{black}{\Sigma}$ Covariance matrix 

* **Diagonal terms**: variance
* **Off-diagonal terms**: correlation
* Properties of Covariance Matrix $\color{black} {\Sigma}$
   1. $\color{black} {\Sigma}$ is Symmetric and Positive Definite.
   2. Diagonalization: $\color{black} {\Sigma}$ can be decomposed in the form of $\color{black}{UDU^T}.$($\color{black}{\rm D}$ is a Diagonal matrix.)   


## MLE of Multivariate Gaussian
 
* Likelyhood:

  $$\color{black}{p({x_i}\mid \mu,\Sigma)}$$
  
  $\color{black}{x_i}$ denotes $\color{black}{ith}$ observation; $\color{black}{\mu,\Sigma}$ are unknown parameters
  
* Objecttive
  
  Estimate the mean and the covariance matrix given obeserved data
  
  $$\color{black}{
  \hat\mu,\hat\Sigma=\arg\max_{\mu.\Sigma}p({x_i}\mid \mu,\Sigma)
  }$$
  
  Instead of maximizing the joint probability, we can assume independence of observations and maximize the product of each probability,
  
  $$\color{black}{
  \hat\mu,\hat\Sigma=\arg\max_{\mu.\Sigma}\prod_{i=1}^N p({x_i}\mid \mu,\Sigma)
  }$$

We can estimate these two parameters analytically. First, we take the log form given the Gaussian distribution density function,

$$\color{black}{
\ln p(x_i\mid \mu,\Sigma)=-\dfrac{1}{2}(x_i-\mu)^T\Sigma^{-1}(x_i-\mu)-\dfrac{1}{2}\ln \vert\Sigma\vert+c
}$$

Then we apply it to the formula above,

$$\color{black}{
  \hat\mu,\hat\Sigma=\arg\max_{\mu.\Sigma}\sum_{i=1}^N \Big\{-\dfrac{1}{2}(x_i-\mu)^T\Sigma^{-1}(x_i-\mu)-\dfrac{1}{2}\ln \vert\Sigma\vert+c \Big\}
  }$$

Finding that the constant term $\color{black}c$ dosen't affect our solution, then we leave out it. As before, we change the formula to minimization problme,

$$\color{black}{
  \hat\mu,\hat\Sigma=\arg\max_{\mu.\Sigma}\sum_{i=1}^N \Big\{\dfrac{1}{2}(x_i-\mu)^T\Sigma^{-1}(x_i-\mu)+\dfrac{1}{2}\ln \vert\Sigma\vert \Big\}
  }$$ 

Contionally we use the cost function $\color{black}{J,}$

$$\color{black}{
J(\mu,\Sigma)=\sum_{i=1}^N \Big\{\dfrac{1}{2}(x_i-\mu)^T\Sigma^{-1}(x_i-\mu)+\dfrac{1}{2}\ln \vert\Sigma\vert \Big\}
}$$

We apply the partial derivative to find the optimum,

1. $\color{black}{\dfrac{\partial J}{\partial\mu}\rightarrow \hat\mu}$

2. $\color{black}{\dfrac{\partial J(\hat\mu,\Sigma)}{\partial\Sigma}\rightarrow \hat\Sigma}$

Finally, we have

$$\color{black}{\begin{align}
\hat\mu&=\dfrac{1}{N}\sum_{i=1}^{N}x_i\\
\hat\Sigma&=\frac{1}{N}\sum_{i=1}{N}(x_i-\hat\mu)(x_i-\hat\mu)^T
\end{align}}$$
































