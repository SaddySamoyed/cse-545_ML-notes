# 1 [23 points] Logistic regression

![Screenshot 2025-02-11 at 15.52.52](hw2.assets/Screenshot 2025-02-11 at 15.52.52.png)

### (a) Find the Hessian of $\ell(w)$

<img src="hw2.assets/image-20250211154432178.png" alt="image-20250211154432178" style="zoom:25%;" />

<img src="hw2.assets/image-20250211154451368.png" alt="image-20250211154451368" style="zoom:30%;" />

<img src="hw2.assets/image-20250211154634735.png" alt="image-20250211154634735" style="zoom:43%;" />



### (b) Prove the $H$ is negative semi-definite![Screenshot 2025-02-11 at 15.45.37](hw2.assets/Screenshot 2025-02-11 at 15.45.37.png)

<img src="hw2.assets/image-20250211154706259.png" alt="image-20250211154706259" style="zoom:43%;" />



### (c) updae rule by Newton's method

![Screenshot 2025-02-11 at 15.45.50](hw2.assets/Screenshot 2025-02-11 at 15.45.50.png)

<img src="hw2.assets/image-20250211154720829.png" alt="image-20250211154720829" style="zoom:15%;" />



![Screenshot 2025-02-11 at 19.31.41](hw2.assets/Screenshot 2025-02-11 at 19.31.41.png)

### (e) coefficients from the code

![Screenshot 2025-02-11 at 15.52.25](hw2.assets/Screenshot 2025-02-11 at 15.52.25.png)

Answer:

<img src="hw2.assets/Screenshot 2025-02-11 at 15.54.17.png" alt="Screenshot 2025-02-11 at 15.54.17" style="zoom:40%;" />



### (f) plot from the code

![Screenshot 2025-02-11 at 15.53.18](hw2.assets/Screenshot 2025-02-11 at 15.53.18.png)

Answer:

<img src="hw2.assets/output.png" alt="output" style="zoom:75%;" />







# 2 [27 points] Softmax Regression via Gradient Ascent

![Screenshot 2025-02-11 at 15.54.57](hw2.assets/Screenshot 2025-02-11 at 15.54.57.png)

![Screenshot 2025-02-11 at 15.55.11](hw2.assets/Screenshot 2025-02-11 at 15.55.11.png)

![Screenshot 2025-02-11 at 15.55.23](hw2.assets/Screenshot 2025-02-11 at 15.55.23.png)



### (a) Derive the gradient ascent update rule for log-likelihood

![Screenshot 2025-02-11 at 15.56.15](hw2.assets/Screenshot 2025-02-11 at 15.56.15.png)![Screenshot 2025-02-11 at 19.19.46](hw2.assets/Screenshot 2025-02-11 at 19.19.46.png)

<img src="hw2.assets/image-20250211175640958.png" alt="image-20250211175640958" style="zoom:42%;" />

<img src="hw2.assets/image-20250211175702303.png" alt="image-20250211175702303" style="zoom:45%;" />



![Screenshot 2025-02-11 at 19.31.09](hw2.assets/Screenshot 2025-02-11 at 19.31.09.png)

### (c) report accuracy

![Screenshot 2025-02-11 at 17.58.14](hw2.assets/Screenshot 2025-02-11 at 17.58.14.png)

<img src="hw2.assets/Screenshot 2025-02-11 at 18.51.02.png" alt="Screenshot 2025-02-11 at 18.51.02" style="zoom:80%;" />





# 3 Gaussian Discriminate Analysis

![Screenshot 2025-02-12 at 13.56.47](hw2.assets/Screenshot 2025-02-12 at 13.56.47.png)

### (a) $p(y=1|x;\phi,\Sigma,\mu_0,\mu_1)$ is a form of logistic function

![Screenshot 2025-02-11 at 19.58.58](hw2.assets/Screenshot 2025-02-11 at 19.58.58.png)



<img src="hw2.assets/image-20250212155747716.png" alt="image-20250212155747716" style="zoom:33%;" />

<img src="hw2.assets/image-20250212155808119.png" alt="image-20250212155808119" style="zoom:33%;" />

### (b) MLE of $\phi,\mu_0,\mu_1$

![Screenshot 2025-02-11 at 20.00.04](hw2.assets/Screenshot 2025-02-11 at 20.00.04.png)

<img src="hw2.assets/image-20250212171037615.png" alt="image-20250212171037615" style="zoom:20%;" />

<img src="hw2.assets/image-20250212171441838.png" alt="image-20250212171441838" style="zoom:33%;" /><img src="hw2.assets/image-20250212171500753.png" alt="image-20250212171500753" style="zoom:18%;" />





<img src="hw2.assets/image-20250213105249367.png" alt="image-20250213105249367" style="zoom:30%;" /><img src="hw2.assets/image-20250213105315270.png" alt="image-20250213105315270" style="zoom:45%;" />





<img src="hw2.assets/image-20250213121355995.png" alt="image-20250213121355995" style="zoom:30%;" /><img src="hw2.assets/image-20250213121418179.png" alt="image-20250213121418179" style="zoom:40%;" />











### (c) MLE of $\Sigma$ when $M=1$

![Screenshot 2025-02-11 at 20.02.48](hw2.assets/Screenshot 2025-02-11 at 20.02.48.png)<img src="hw2.assets/image-20250213124648657.png" alt="image-20250213124648657" style="zoom:38%;" /><img src="hw2.assets/image-20250213124743239.png" alt="image-20250213124743239" style="zoom:20%;" />











### (d) MLE of $\Sigma$ when $M>1$ 

![Screenshot 2025-02-11 at 20.03.29](hw2.assets/Screenshot 2025-02-11 at 20.03.29.png)

<img src="hw2.assets/image-20250213140106599.png" alt="image-20250213140106599" style="zoom: 40%;" />

<img src="hw2.assets/image-20250213140229519.png" alt="image-20250213140229519" style="zoom: 43%;" />















# 4 Naive Bayes for Classifying SPAM

### (a) Naive Bayes with Bayesian Smoothing

![Screenshot 2025-02-11 at 19.40.56](hw2.assets/Screenshot 2025-02-11 at 19.40.56.png)

<img src="hw2.assets/image-20250213181957281.png" alt="image-20250213181957281" style="zoom:50%;" />

<img src="hw2.assets/image-20250213182010534.png" alt="image-20250213182010534" style="zoom:50%;" />

<img src="hw2.assets/image-20250213182022870.png" alt="image-20250213182022870" style="zoom:45%;" />















### (b) SPAM classifier

![Screenshot 2025-02-11 at 19.44.33](hw2.assets/Screenshot 2025-02-11 at 19.44.33.png)

![Screenshot 2025-02-11 at 19.44.43](hw2.assets/Screenshot 2025-02-11 at 19.44.43.png)

   - For each email \(\mathbf{x}\), we compute:
     $$
     \log P(\text{spam} \mid \mathbf{x}) \approx \log \phi + \sum_j \bigl(\mathbf{x}[j] \cdot \log(\mu_{\text{spam}}[j])\bigr)
     $$
     $$
     \log P(\text{nonspam} \mid \mathbf{x}) \approx \log(1-\phi) + \sum_j \bigl(\mathbf{x}[j] \cdot \log(\mu_{\text{nonspam}}[j])\bigr)
     $$
   - We label the email as spam (`1`) if $log P(\text{spam} \mid \mathbf{x}) > \log P(\text{nonspam} \mid \mathbf{x})$, otherwise non-spam (`0`).



![Screenshot 2025-02-11 at 19.45.28](hw2.assets/Screenshot 2025-02-11 at 19.45.28.png)<img src="hw2.assets/Screenshot 2025-02-13 at 18.22.05.png" alt="Screenshot 2025-02-13 at 18.22.05" style="zoom:80%;" />



![Screenshot 2025-02-11 at 19.45.36](hw2.assets/Screenshot 2025-02-11 at 19.45.36.png)

report is as below:

<img src="hw2.assets/Screenshot 2025-02-13 at 18.24.03.png" alt="Screenshot 2025-02-13 at 18.24.03" style="zoom:80%;" />







![Screenshot 2025-02-11 at 19.45.46](hw2.assets/Screenshot 2025-02-11 at 19.45.46.png)

<img src="hw2.assets/image-20250213182450737.png" alt="image-20250213182450737" style="zoom:60%;" />









![Screenshot 2025-02-11 at 19.45.54](hw2.assets/Screenshot 2025-02-11 at 19.45.54.png)

Answer: 1400 mail data training set provides the best classification accuracy of 98.375%. By looking at our optimal solution of $\mu_i^j = \frac{N_{C_i}^j + \alpha}{\sum_{j'=1}^{M} (N_{C_i}^{j'} + \alpha)}$ and $ \phi_i = \frac{N_{C_i}}{\sum_{i'} N_{C_{i'}}}$, we can see that the classification is a direct reflection of how frequent a word in one email occurs in spam/nonspam emails. So more data gives more generality. 











