---
layout: post_layout
title: Softmax Regression
time: 2016年08月04日 星期四
location: 南京
pulished: true
excerpt_separator: "```"
---

$$$\large\qquad\qquad\qquad

   h_\theta(x)=

   \begin{bmatrix} P(y=1|x;\theta\\P(y=2|x;\theta\\\vdots\\P(y=K|x;\theta)\end{bmatrix}

   =\dfrac{1}{\sum_{j=1}^K{\rm{exp}}(\theta^{(j){\rm{T}}}x)}

   \begin{bmatrix} {\rm{exp}}(\theta^{(1){\rm{T}}}x)\\

   {\rm{exp}}(\theta^{(2){\rm{T}}}x)\\ \vdots\\

   {\rm{exp}}(\theta^{(K){\rm{T}}}x)

   \end{bmatrix}$$$
