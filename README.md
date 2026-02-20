# Artificial Intelligence & Machine Learning Repository

This repository contains a comprehensive collection of AI and Machine Learning implementations, ranging from fundamental algorithms to advanced agents and MLOps pipelines. It serves as a codebase for exploring various domains of AI including Agents, LLMs, Computer Vision, and Reinforcement Learning.

## 🤖 AI Agents & Orchestration

### Autonomous Agents

- **Frameworks**: AutoGPT, CrewAI, AutoGen, LangGraph.
- **Key Concepts**: Multi-agent orchestration, autonomous task execution, tool use, and cognitive architectures.
- **Implementations**:
  - **CrewAI**: Role-based agent teams for complex workflows.
  - **LangGraph**: Stateful, graph-based agent orchestration.
  - **AutoGen**: Conversational patterns between multiple agents.

### LangChain

- **Basics**: Foundational scripts and chains using the LangChain framework.
- **Pipelines**: RAG (Retrieval-Augmented Generation) and custom chains.

---

## 🧠 Machine Learning Algorithms

### Supervised Learning

- **Regression**: Linear, Polynomial, Ridge/Lasso.
- **Classification**: Logistic Regression, SVM, Decision Trees, Random Forest, Naive Bayes, KNN.
- **Gradient Boosting**: Implementations of boosting algorithms.

### Unsupervised Learning

- **Clustering**: K-Means, DBSCAN, Hierarchical Clustering, GMM (Gaussian Mixture Models).
- **Dimensionality Reduction**: PCA (Principal Component Analysis), Autoencoders.
- **Anomaly Detection**: Isolation Forest, One-Class SVM.

### Deep Learning & Neural Networks

- **Architectures**: CNNs, RNNs, LSTMs.
- **Transformers**: Implementation of attention mechanisms and transformer blocks.
- **Self-Training**: Semi-supervised learning approaches.

### Reinforcement Learning

- **Value-Based**: Q-Learning, Deep Q-Networks (DQN).
- **Policy-Based**: Policy Gradient methods.

---

## 🛠️ MLOps & Engineering

### Deployment & Infrastructure

- **Docker**: Containerized ML model training and inference environments.
- **Kubernetes**: Local deployment configurations (`deployment.yaml`, `service.yaml`) for scaling ML services.
- **Path to Production**: Notebooks and scripts demonstrating the lifecycle of moving models from research to production.

---

## 🚀 Applied Projects

### DeepSeek Integrations

A suite of practical tools built using DeepSeek LLMs:

- **Developer Tools**: Code Autocompletion, Debugger, Documentation Generator.
- **Content & Analysis**: Financial Report Summarizer, Feedback Analyzer, Content Writer.
- **Assistants**: Legal Assistant, Medical Symptom Checker, Job Application Screener.

### Qwen & Ollama

- Experiments and implementations using Qwen models and Ollama for local LLM inference.

---

## 📚 Frameworks & Libraries

- **PyTorch**: Deep learning implementations using torch tensors and autograd.
- **TensorFlow**: Neural network models built with TF/Keras.
- **IBMBee**: JavaScript-based agent framework experiments.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js (for JS-based agents)
- Docker (for MLOps)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd AI
   ```

2. **Set up Python Environment**

   ```bash
   python -m venv AI_env
   source AI_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Explore Agents**
   Navigate to `AIAgents/` to run specific agent frameworks.

---

## Tech Stack

- **Languages**: Python, JavaScript
- **ML Frameworks**: PyTorch, TensorFlow, Scikit-learn
- **LLM Frameworks**: LangChain, AutoGen, CrewAI, Ollama
- **Infrastructure**: Docker, Kubernetes
