# Attention-mechanism-from-scratch

# Self-Attention Mechanism Using NumPy

## Overview
This project demonstrates the implementation of the **Self-Attention Mechanism** using Python and NumPy. Self-attention is one of the core concepts used in Transformer models, allowing each input token to focus on the most relevant tokens while generating its representation.

The implementation covers the complete workflow from input embeddings to the final attention output.

---

## Objective
- Understand the working of the Self-Attention mechanism.
- Learn how Query (Q), Key (K), and Value (V) matrices are generated.
- Calculate attention scores and attention weights using Softmax.
- Produce the final attention output.

---

## Technologies Used
- Python 3.x
- NumPy

---

## Input Embeddings

```python
X = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 2.0, 1.0],
    [1.0, 1.0, 0.0]
])
```

---

## Steps Performed

### 1. Generate Random Weight Matrices
- Query Weight Matrix (WQ)
- Key Weight Matrix (WK)
- Value Weight Matrix (WV)

### 2. Compute Query, Key and Value Matrices

```
Q = X × WQ
K = X × WK
V = X × WV
```

### 3. Compute Attention Scores

```
Attention Scores = Q × Kᵀ
```

### 4. Scale the Scores

```
Scaled Scores = Scores / √dₖ
```

where **dₖ** is the key dimension.

### 5. Apply Softmax

```
Attention Weights = Softmax(Scaled Scores)
```

### 6. Generate Final Attention Output

```
Attention Output = Attention Weights × V
```

---

## Output

### Attention Weights

```
[[0.20444479 0.54498085 0.25057436]
 [0.28246035 0.49925538 0.21828427]
 [0.19052918 0.59425076 0.21522006]]
```

### Final Attention Output

```
[[0.53313706 0.72681841 1.48712448]
 [0.54720320 0.72450887 1.47765568]
 [0.52712059 0.74060265 1.52119990]]
```

---

## Project Structure

```
Self-Attention/
│
├── self_attention.py
├── README.md
```

---

## Key Concepts

- Input Embeddings
- Query (Q)
- Key (K)
- Value (V)
- Attention Scores
- Scaled Dot-Product Attention
- Softmax Function
- Attention Weights
- Transformer Fundamentals

---

## Formula

```
Attention(Q,K,V) = Softmax((QKᵀ)/√dₖ)V
```

---

## Learning Outcome

After completing this project, you will understand:

- How Self-Attention works internally.
- The role of Query, Key, and Value matrices.
- Why scaling is necessary.
- How Softmax converts scores into probabilities.
- How the final contextual representation is generated.

---
