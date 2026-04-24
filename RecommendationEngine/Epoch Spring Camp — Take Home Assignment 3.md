# **Epoch Spring Camp — Take Home Assignment 3**

## **Building a Neural Recommender System**

---

## **Objective**

In this assignment, you will build a recommender system and explore how deep learning can model complex user behavior.

You will start with a simple baseline model and then improve it using a Multi-Layer Perceptron (MLP) inspired by modern recommender systems.

---

## **Background**

Recommender systems are used in platforms like Netflix, Spotify, and Amazon to suggest relevant content.

A common approach is Collaborative Filtering, where we learn from user–item interactions.

Traditionally, these systems use a dot product between user and item vectors. In this assignment, you will:

* Implement this baseline  
* Replace it with a neural network (MLP)  
* Compare their performance

---

## **What You Should Read**

You are not expected to know about recommender systems beforehand. The following topics are sufficient to complete the assignment.

### **1\. Embeddings**

* What is an embedding layer  
* How discrete IDs (user/item) are mapped to vectors  
* Why embeddings are learned

Focus: intuition over theory

### **2\. Matrix Factorization (Basic Idea)**

* Representing users and items as vectors  
* Dot product as a similarity score

You do not need full mathematical depth—just understand how predictions are made.

### **3\. Multi-Layer Perceptron (MLP)**

* Linear layers  
* Activation functions (ReLU, Sigmoid)  
* Forward pass structure

### **4\. Binary Classification with Neural Networks**

* Output as probability (0–1)  
* Binary Cross Entropy loss  
* Interpreting predictions

### **5\. Implicit Feedback (Important)**

* Difference between explicit vs implicit data  
* Why we treat interactions as 1 and sample negatives as 0

### **6\. Basic PyTorch Usage**

* Defining a model (`nn.Module`)  
* Using embeddings (`nn.Embedding`)  
* Training loop (forward, loss, backward, step)

### **Optional (for deeper understanding)**

* Ranking metrics (Hit@K, NDCG)  
* Overfitting and regularization

---

## **Task Overview**

### **Task 1: Baseline Model (Matrix Factorization)**

* Represent each user and item as an embedding vector  
* Predict interaction using dot product  
* Train the model on the given dataset

### **Task 2: Neural Model (MLP-based)**

* Replace the dot product with an MLP  
* Concatenate user and item embeddings  
* Pass them through a neural network to predict interaction

### **Task 3: Evaluation and Comparison**

* Compare baseline vs MLP model  
* Use the provided evaluation metric  
* Analyze:  
  * Which performs better  
  * Why that might be the case

### **Optional Experiments**

Try improving your model. Examples:

* Change number of layers  
* Try different activation functions  
* Tune embedding size  
* Combine dot product and MLP

**Bonus Task**

* Combine the dot product and MLP approaches\!

---

## **Technical Requirements**

* Use PyTorch  
* You may refer to documentation and conceptual resources  
* Do not copy full implementations from external sources

---

## **Get Started\!**

* [Dataset and Starter notebook](https://drive.google.com/drive/folders/17_bkT410kg-HP-L9ICcqYhiMsw0YOFPN?usp=drive_link)  
* Resources to get you started:  
  * [Neural Network Embeddings Explained](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526/)  
  * [Negative Sampling in Recommender Systems \- Read for Intuition\!](https://www.bcg.com/x/the-multiplier/explaining-negative-sampling-in-recommender-systems)  
  * [Pytorch Documentation](https://docs.pytorch.org/docs/stable/index.html)  
  * [Neural Collaborative Filtering \- Paper this task is based on](https://arxiv.org/pdf/1708.05031)

---

## **Evaluation Metric**

You may use:

* Hit@K (used for recommendation tasks)  
* Accuracy (Optional)

Details will be provided in the notebook.

---

## **Deliverables**

1. Code  
   * Clean, readable implementation  
2. Short Report (1–2 pages)  
   Include:  
   * Your approach  
   * Key design choices  
   * Results (baseline vs MLP)  
   * What you tried beyond the basics  
   * Observations

---

## **Timeline**

Deliverables are expected to be submitted by May 2nd, 2026\.