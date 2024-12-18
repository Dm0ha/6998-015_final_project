# Expediting Model Evaluation Through Lexical Simplification

This repository contains the code, experiments, and documentation for our COMSE6998 GenAI project. Our project introduces a novel approach to evaluate large language model (LLM) architectures using simplified English datasets, reducing computational costs and providing insights into architecture performance differences.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Objectives](#project-objectives)
3. [Project Structure](#project-structure)
4. [User guide](#user-guide)
5. [References](#references)

---

## Introduction

Large language models like GPT and BERT have transformed NLP but require significant computational resources to train. Our project simplifies English datasets using word2vec embeddings to reduce vocabulary size, enabling faster and cheaper evaluations of LLM architectures. This method allows researchers to test architectural changes with reduced computational constraints, providing a cost-effective and scalable solution.

### Team：
- Luke Tingley (lt2985)
- Yuehui Ruan (yr2453)

## Project Objectives

### Key Goals
- **Simplified English Dataset Creation**: Use word2vec to merge semantically similar words.
- **LLM Architecture Evaluation**: Assess architecture performance using simplified datasets.
- **Comparative Analysis**: Test whether performance differences on simplified datasets predict differences on full datasets.

### Expected Outcomes
- Evidence supporting or refuting the effectiveness of simplified datasets for model evaluation.
- Insights into optimal vocabulary reduction techniques and their impact on model scaling.


## Project Structure

### Prerequisites
- Python >= 3.8
- pip or conda for package management
- Access to GPU/TPU for training (e.g., GCP, AWS)

### Prepare the dataset and pre-trained word2vec embeddings:
- Download word2vec embeddings.
- Place them in the data/ directory.

## User guide


## References

- **Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)**  
  [Link to Paper](https://arxiv.org/abs/1301.3781)

- **Attention Is All You Need (Vaswani et al., 2017)**  
  [Link to Paper](https://arxiv.org/abs/1706.03762)

- **BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)**  
  [Link to Paper](https://arxiv.org/abs/1810.04805)

- **TinyStories Dataset (Eldan et al., 2023)**  
  [Link to Paper](https://arxiv.org/abs/2305.07759)
