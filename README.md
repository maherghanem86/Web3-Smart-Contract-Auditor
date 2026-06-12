🛡️ Web3 Smart Contract Security Auditor

This repository contains the code and resources for an advanced AI-powered Smart Contract vulnerability detection system. The project introduces the Triple Fusion Sentinel architecture, a novel multi-modal framework that combines static code analysis, sequential logic tracking, and expert-crafted statistical features to achieve highly accurate vulnerability severity classification in Solidity smart contracts.

🌟 Key Features

Multi-Modal Fusion Architecture: The system employs a unique Triple Attention mechanism to dynamically weight and combine three distinct data modalities:

Static Embeddings: Extracted using Microsoft's graphcodebert-base to capture data flow and Abstract Syntax Trees (AST).

Dynamic/Sequential Embeddings: Extracted using web3se/SmartBERT-v3 to process the sequential and behavioral nature of contract logic and intent.

Expert Features: Statistical measures (mean, std, max, norm) indicating structural integrity.

Focal Loss & SMOTE: Implements Synthetic Minority Over-sampling Technique (SMOTE) and Focal Loss to address severe class imbalances in the dataset, forcing the model to focus on hard-to-classify, high-severity vulnerabilities (e.g., Reentrancy, Incorrect Interfaces).

Interactive Web Interface: Fully deployed on Hugging Face Spaces using Gradio, providing a user-friendly code editor to instantly audit smart contracts.

🛠️ System Architecture

The core of the system relies on the TripleFusionSentinel neural network:

Independent feed-forward pathways with batch normalization and high dropout rates (to prevent overfitting) process each modality.

A Tanh-Sigmoid attention mechanism fuses the features, determining the relevance of static vs. dynamic vs. expert cues.

The output classifies the severity into one of four categories: High, Medium, Low, None.

🚀 Live Demo

Experience the AI auditor directly in your browser. The demo includes predefined tests for classic vulnerabilities, including Reentrancy and Incorrect Interface mappings (e.g., Crytic's Alice.sol):
👉 Launch Web3 Smart Contract Auditor
https://huggingface.co/spaces/maherghanem86/Web3-Smart-Contract-Auditor?logs=build

Local Execution

To run the Gradio app locally:

Clone the repository.

Install dependencies: pip install -r requirements.txt


Run the application: python app.py (The script will automatically fetch the pre-trained weights, GraphCodeBERT, and SmartBERT tokenizers from Hugging Face).


Model URL:  https://huggingface.co/maherghanem86/Web3-Smart-Contract-Auditor

Dataset  URL:   https://huggingface.co/datasets/maherghanem86/Solidity-Dataset 
