# UCB-OCO Hybrid Optimizer for Recommender Systems

### Numerical Optimization Mini Project – Theme 9

**Logarithmic Regret Bounding via UCB in Collaborative Filtering**

---

## Project Overview

Recommender systems are widely used in modern digital platforms to help users discover relevant content among large collections of items. A key challenge in recommender systems is balancing **exploration** (discovering new items) and **exploitation** (recommending items already known to be relevant).

Traditional collaborative filtering approaches, such as **Matrix Factorization trained with Stochastic Gradient Descent (SGD)**, focus primarily on prediction accuracy and do not explicitly address exploration.

This project proposes a **hybrid optimization framework** that combines:

* **Upper Confidence Bound (UCB)** – a bandit-based exploration strategy
* **Online Convex Optimization (OCO)** – adaptive learning using gradient-based updates

The system models recommendation as a **multi-armed bandit problem**, where each user–item interaction represents a decision with a reward signal. The UCB mechanism dynamically encourages exploration of uncertain recommendations while maintaining strong prediction performance.

---

## Dataset

The experiments use the **MovieLens 100K dataset**, which contains:

* **943 users**
* **1682 movies**
* **100,000 ratings**

Ratings range from **1 to 5**, representing user preferences for movies.

Dataset source:
https://grouplens.org/datasets/movielens/100k/

Dataset structure expected in the project:

```
project_folder/
│
├── ml-100k/
│   └── u.data
│
├── results/
│
└── ucb_oco_recommender.py
```

---

## Methodology

### 1. Collaborative Filtering

The recommendation system is based on **Matrix Factorization**, where the user–item interaction matrix is decomposed into latent user and item vectors.

Predicted rating:

```
r̂_ui = μ + b_u + b_i + U_u^T V_i
```

Where:

* μ = global mean rating
* b_u = user bias
* b_i = item bias
* U, V = latent factor matrices

---

### 2. UCB Exploration

Each user–item pair is treated as a **bandit arm**.

The **Upper Confidence Bound (UCB)** score balances expected reward and uncertainty:

```
UCB(u,i) = μ̂(u,i) + α σ̂(u,i) / √(N(u,i) + 1)
```

Where:

* μ̂(u,i) = empirical mean reward
* σ̂(u,i) = empirical variance
* N(u,i) = number of interactions
* α = exploration coefficient

This encourages the model to explore **less-observed items**, improving cold-start recommendations.

---

### 3. Online Convex Optimization (OCO)

Model parameters are updated using **AdaGrad**, which adapts learning rates based on historical gradients.

AdaGrad update rule:

```
θ_{t+1} = θ_t − η * g_t / √(G_t + ε)
```

Where:

* g_t = gradient at time t
* G_t = accumulated squared gradients
* η = base learning rate

This allows stable optimization in sparse recommendation datasets.

---

## Evaluation Metrics

The system is evaluated using multiple metrics:

| Metric            | Description                                           |
| ----------------- | ----------------------------------------------------- |
| RMSE              | Prediction error between predicted and actual ratings |
| Precision@K       | Relevance of recommended items                        |
| Coverage          | Diversity of recommended items                        |
| Cumulative Regret | Efficiency of exploration                             |

---

## Experimental Results

| Metric        | UCB-OCO Model | SGD Baseline |
| ------------- | ------------- | ------------ |
| RMSE          | **0.9360**    | 0.9446       |
| Precision@K   | **0.76**      | 0.66         |
| Coverage@10   | 0.100         | 0.113        |
| Training Time | 15.4 s        | 8.3 s        |

### Key Observations

* The **UCB-OCO model achieves higher Precision@K**, indicating better recommendation relevance.
* The method improves **exploration capability** compared to traditional SGD.
* Training convergence remains stable due to **AdaGrad optimization**.

---

## Generated Outputs

Running the code produces several evaluation plots:

```
results/
│
├── fig1_convergence.png
├── fig2_regret.png
├── fig3_precision.png
├── fig4_dashboard.png
└── metrics.json
```

These visualizations show:

* RMSE convergence
* cumulative regret
* precision comparison
* overall performance dashboard

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/ucb-oco-recommender.git
cd ucb-oco-recommender
```

Install dependencies:

```
pip install numpy pandas matplotlib
```

---

## Running the Project

Place the **MovieLens dataset** inside the `ml-100k` folder and run:

```
python ucb_oco_recommender.py
```

The program will:

1. Load the MovieLens dataset
2. Train the **UCB-OCO model**
3. Train the **SGD baseline**
4. Evaluate performance metrics
5. Generate visualizations and results

---

## Project Structure

```
ucb-oco-recommender/
│
├── ml-100k/
│   └── u.data
│
├── results/
│
├── ucb_oco_recommender.py
├── README.md
└── report.pdf
```

---

## Team Members

| USN        | Name                |
| ---------- | ------------------- |
| 23BTRCL131 | Aryan Agrawal       |
| 23BTRCL003 | Yathin Girish Kumar |
| 23BTRCL244 | Akash SM            |
| 23BTRCL137 | Druvraaj            |

Department of **Artificial Intelligence and Machine Learning**
Faculty of Engineering and Technology

---

## Project Guide

Dr. Shivam Swarup
Assistant Professor – AIML
Domain: **Online Optimization, Bandits & Distributed AI**

---

## Conclusion

This project demonstrates that integrating **bandit-based exploration (UCB)** with **online optimization (OCO)** provides an effective framework for collaborative filtering recommender systems. The hybrid approach improves exploration efficiency while maintaining strong prediction performance, making it a promising method for modern recommendation platforms.

---

## Future Work

Possible future improvements include:

* Applying the method to **larger recommender datasets**
* Incorporating **context-aware recommendations**
* Exploring **deep learning–based recommendation models**
* Improving exploration strategies with **advanced bandit algorithms**

---

## License

This project was developed as part of a **Numerical Optimization course mini-project** and is intended for academic use.
