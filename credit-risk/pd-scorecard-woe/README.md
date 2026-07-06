# Probability of Default Scorecard with WOE/IV Logistic Regression

## 1. Problem Statement
Application and behavioural scorecards are the workhorse of retail credit. Regulators
and model-risk teams expect them to be **transparent, monotonic and auditable** — which
is why the Weight-of-Evidence (WOE) + logistic regression recipe has survived decades of
fancier alternatives. The craft is in the binning, not the algorithm.

**Business Impact:**
* **Approvals:** A better rank-ordering model approves more good customers at the same loss.
* **Compliance:** WOE scorecards give per-feature, points-based explanations out of the box.

## 2. Solution Overview
We build a full scorecard: **WOE binning** of raw features, **Information Value (IV)**
feature screening, a logistic regression on WOE-transformed inputs, and **points scaling**
(PDO/base-odds). Evaluation uses the credit-standard **KS statistic and Gini**, not accuracy.

## 3. Techniques & Features
* **WOE = ln(%good / %bad)** per bin — linearises the relationship and handles outliers/missing.
* **IV** ranks predictive power (rules of thumb: <0.02 useless, >0.5 suspiciously strong).
* **Points = Offset + Factor × (β·WOE)** — the deployable scorecard.

**Algorithm:** WOE/IV + Logistic Regression scorecard, KS/Gini validation.
**Libraries:** Scikit-Learn, Pandas, Numpy, Matplotlib.

## 4. Repository Structure
* `credit_applications.csv`: Synthetic application dataset with default flags.
* `PD_Scorecard_WOE.ipynb`: Binning, IV, scorecard fit, KS/Gini.
* `requirements.txt`: Dependencies.
