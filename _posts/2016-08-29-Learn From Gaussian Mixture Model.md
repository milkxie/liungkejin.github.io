---
layout: post_layout
title: Learn form Gaussian Mixture Model
time: 2016.8.29 Monday
location: Nanjing
pulished: true
excerpt_separator: "#"
---

# Gaussian Mixture Model


GMM can be written as a weighted sum of Gaussians,

$$\color{black}{
p(x)=\sum\limits_{k=1}^Kw_kg_k(x\mid \mu_k,\Sigma_k)
}$$

$\color{black}{g_k}$ denotes a single Gaussian density of $\color{black}{\mu_k}$ and $\color{black}{\Sigma_k,}$ and $\color{black}{K}$ denote the number of Gaussian components.

where,

$$\color{black}{
\sum_{k=1}^Kw_k=1, w_k>0 
}$$



Theoretically, a arbitrarily large K and small variances can lead to any shape of distribution.

The largest disadvantage of GMM is that it hard to estimate the parameters when there are a enumerous number of parameters. We can not analytically find the solution for  the GMM parameters.   

## GMM Parameter Estimation via EM

### Learning GMM Parameters

The objective 

$$\color{black}{
\hat\mu,\hat\Sigma=\arg\max_{\mu,\Sigma}\prod_{i=1}^Np(x_i\mid \mu,\Sigma)
}$$

Take the log and apply the Gaussian Mixture Model, we have

$$\color{black}{
\hat\mu,\hat\Sigma=\arg\max_{\mu,\Sigma}\sum_{i=1}^N\ln\Big\{\dfrac{1}{K}\sum_{k=1}^K g_k(x_i\mid u_k,\Sigma_k)\Big\}
}$$

where,

$$\color{black}{
g_k(x)=\dfrac{1}{(2\pi)^{D/2}\vert\Sigma\vert^{1/2}}\exp\Big\{-\dfrac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)\Big\} 
}$$

Note that we use the uniform weights $ \color{black}{1/K}$ and focus on understanding how to estimate the mean and covariance matrix parameters.

It turns out that we cannot solve the problem analytically. This implies we can estimate the parameters iteratively.

## Expectation-Maximization(EM) Algorithm

As follow diagram, we can easily get the inequality equation $\color{black}{f(\dfrac{x_1+x_2}{2})\le\dfrac{f(x_1)+f(x_2)}{2}}$ , the special case of the Jensen's inequality.

<img src="/assets/img/Jensen_inequality.png" width="400px" />

Now we generalize this idea to any posotively weighted cases or multiple points as long as the function is convex. We have

$$\color{black}{
f\Bigg(\sum a_ix_i\Bigg)\le\sum a_i f(x_i)
}$$

where,

$$\color{black}{\begin{align}
\sum a_i&=1\\
a_i&\ge 0
\end{align}}$$

When the function is concave, we have

$$\color{black}{
f\Bigg(\sum a_ix_i\Bigg)\ge\sum a_i f(x_i)
}$$

where,

$$\color{black}{\begin{align}
\sum a_i&=1\\
a_i&\ge 0
\end{align}}$$

The function we use is a log function, also a concave function.

Now we introduce the latent variable $\color{black}{z,}$

$$\color{black}{
p(X\mid \theta) = \sum_Z p(X,Z\mid \theta)
}$$

It is a marginal probability over the variable. As before we take the log for the above formula, and then apply the Jensen's inequality. we get a lower broud of the log-likelihood,

$$\color{black}{\begin{align}
\ln p(X\mid \theta) &=\ln \sum_Z p(X,Z\mid \theta)\\
&=\ln\sum_Z q(Z)\dfrac{p(X,Z\mid\theta)}{q(Z)}\ge\sum_Z q(Z)\ln\dfrac{p(X,Z\mid\theta)}{q(Z)} 
\end{align}}$$

Note that $\color{black}{q(Z)}$ is a valid probability distribution over $\color{black}{Z.}$ Now we describe the process for finding the local maximum likelihood solutions of $\color{black}{F}$ using lower bound  $\color{black}{G}$ .

1. Find a lower bound  $\color{black}{G}$ with an initial guess  
$$\color{black}{\ln p(X\mid \theta)\ge \sum\limits_Z q(Z)\ln\dfrac{p(X,Z\mid\theta)}{q(Z)}}$$  

2. Find $\color{black}{\theta^* = \arg\max\limits_\theta G(\theta\mid\theta_0)}$  
<img src="/assets/img/EM_algorithm1.png" width="400px" />

3. Find a new lower bound $\color{black}{G}$ with $\color{black}{\theta_1\leftarrow\theta^*}$  
<img src="/assets/img/EM_algorithm2.png" width="400px" />

4. Find $\color{black}{\theta^*=\arg\max\limits_\theta G(\theta\mid\theta_1)}$  
<img src="/assets/img/EM_algorithm3.png" width="400px" />

5. Repeat until converged  
<img src="/assets/img/EM_algorithm4.png" width="400px" />




























