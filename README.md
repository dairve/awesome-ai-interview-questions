# awesome-ai-interview-questions

> A curated collection of AI, Machine Learning, and LLM interview questions for 2025. Perfect for developers preparing for FAANG interviews.

[![GitHub stars](https://img.shields.io/github/stars/codiebyheaart/awesome-ai-interview-questions.svg?style=social&label=Star)](https://github.com/codiebyheaart/awesome-ai-interview-questions)
[![GitHub forks](https://img.shields.io/github/forks/codiebyheaart/awesome-ai-interview-questions.svg?style=social&label=Fork)](https://github.com/codiebyheaart/awesome-ai-interview-questions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Table of Contents

- [AI & Machine Learning](#ai--machine-learning)
- [Large Language Models (LLMs)](#large-language-models)
- [Deep Learning](#deep-learning)
- [Natural Language Processing](#nlp)
- [Computer Vision](#computer-vision)
- [MLOps & Deployment](#mlops)
- [System Design for AI](#system-design)
- [Coding Challenges](#coding-challenges)

---

## 🤖 AI & Machine Learning

### Q1: Explain the difference between supervised and unsupervised learning

**Answer:**
- **Supervised Learning:** Uses labeled data to train models. Examples: classification, regression
  - Example: Email spam detection (labeled: spam/not spam)
  
- **Unsupervised Learning:** Finds patterns in unlabeled data
  - Example: Customer segmentation, clustering

**When to use:**
- Supervised: When you have labeled training data
- Unsupervised: For exploratory analysis, anomaly detection

**Code Example:**
````python
# Supervised Learning
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)  # Labels required

# Unsupervised Learning
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)  # No labels needed
````

---

### Q2: What is overfitting and how do you prevent it?

**Answer:**
Overfitting occurs when a model learns training data too well, including noise, and performs poorly on new data.

**Prevention techniques:**
1. **Cross-validation** - Split data into train/validation/test
2. **Regularization** - L1 (Lasso) or L2 (Ridge)
3. **Early stopping** - Stop training when validation loss increases
4. **Dropout** - Randomly disable neurons during training
5. **Data augmentation** - Increase training data variety

**Code Example:**
````python
from sklearn.linear_model import Ridge

# L2 Regularization
model = Ridge(alpha=1.0)  # alpha controls regularization strength
model.fit(X_train, y_train)

# Dropout in Neural Networks (Keras)
from tensorflow.keras.layers import Dropout
model.add(Dropout(0.5))  # Drop 50% of neurons
````

---

## 🧠 Large Language Models (LLMs)

### Q3: How do transformer models work?

**Answer:**
Transformers use **self-attention mechanisms** to process sequences in parallel (unlike RNNs).

**Key Components:**
1. **Self-Attention:** Weighs importance of different words
2. **Multi-Head Attention:** Multiple attention mechanisms in parallel
3. **Positional Encoding:** Adds position information
4. **Feed-Forward Networks:** Process each position

**Example Architecture:**

---

## 🔍 Natural Language Processing

### Q5: What is the difference between BERT and GPT?

**Answer:**

| Feature | BERT | GPT |
|---------|------|-----|
| **Architecture** | Encoder-only | Decoder-only |
| **Training** | Masked Language Model | Autoregressive |
| **Use Case** | Understanding (classification) | Generation (text completion) |
| **Bidirectional** | Yes | No (left-to-right) |

**When to use:**
- **BERT:** Sentiment analysis, Q&A, classification
- **GPT:** Text generation, chatbots, creative writing

---

## 💻 Coding Challenges

### Q6: Implement a simple neural network from scratch

**Answer:**
````python
import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Usage
nn = SimpleNN(input_size=2, hidden_size=4, output_size=1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

# Train
for epoch in range(10000):
    output = nn.forward(X)
    nn.backward(X, y)

print("Predictions:", nn.forward(X))
````

---

## 🏗️ System Design for AI

### Q7: Design a scalable ML inference system

**Answer:**

**Architecture:**

**Key Considerations:**
1. **Latency:** Use model caching, batch inference
2. **Scaling:** Horizontal scaling with Kubernetes
3. **Model versioning:** A/B testing, blue-green deployment
4. **Monitoring:** Track accuracy drift, latency, errors
5. **Fallback:** Have a simpler backup model

**Code Example (FastAPI):**
````python
from fastapi import FastAPI
import torch

app = FastAPI()

# Load model once at startup
model = torch.load('model.pt')
model.eval()

@app.post("/predict")
async def predict(data: dict):
    with torch.no_grad():
        prediction = model(data['input'])
    return {"prediction": prediction.tolist()}
````

---

## 🎓 Study Resources

- [CS229: Machine Learning - Stanford](http://cs229.stanford.edu/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Hugging Face NLP Course](https://huggingface.co/course)
- [Papers with Code](https://paperswithcode.com/)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/new-question`)
3. Commit changes (`git commit -am 'Add new question'`)
4. Push to branch (`git push origin feature/new-question`)
5. Open a Pull Request

**Contribution Guidelines:**
- Add questions with detailed answers
- Include code examples
- Cite sources when applicable
- Keep explanations clear and concise

---

## 📬 Contact

- **GitHub:** [@codiebyheaart](https://github.com/codiebyheaart)


---

## ⭐ Support

If this helped you, please:
- ⭐ Star this repository
- 🔀 Fork and contribute
- 📢 Share with others

---

## 📝 License

MIT License - feel free to use for interviews, learning, or teaching!

---

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/codiebyheaart/awesome-ai-interview-questions)
![GitHub forks](https://img.shields.io/github/forks/codiebyheaart/awesome-ai-interview-questions)
![GitHub issues](https://img.shields.io/github/issues/codiebyheaart/awesome-ai-interview-questions)
![GitHub contributors](https://img.shields.io/github/contributors/codiebyheaart/awesome-ai-interview-questions)

**Last Updated:** December 2025

---

<p align="center">Made with ❤️ by developers, for developers</p>
