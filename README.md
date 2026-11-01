# Dual-Engine Spatio-Temporal Deep Learning with Deterministic Guardrails for Aviation Delay Propagation

This repository contains the processed datasets, PyTorch models, and the 3-Tier Streamlit MLOps deployment framework for the manuscript "Dual-Engine Spatio-Temporal Deep Learning with Deterministic Guardrails for Aviation Delay Propagation".

## Overview
Cascading flight delays cost the global aviation industry billions annually. Traditional tabular machine learning models (like XGBoost) fail to capture physical network topology, while deep Spatio-Temporal Graph Neural Networks (ST-GNNs) suffer from Graph Over-Smoothing, often erasing catastrophic "Black Swan" delay events.

This project introduces a Dual-Engine Architecture that encapsulates an Optimized Long Short-Term Memory Graph Convolutional Network (LSTGCN) within a deterministic Business Rules Engine. This creates a transparent, Human-In-The-Loop (HITL) decision-support system that alerts dispatchers to extreme delays (>180 minutes) before the neural network mathematically dilutes them.

## Repository Structure
data/: Contains the extracted dataset derived from the U.S. DOT aviation records used for model training and evaluation.

## Data Availability
The raw historical flight records were sourced from the public United States Department of Transportation (DOT) Bureau of Transportation Statistics repository.
The files provided in the data/ directory of this repository represent the extracted dataset used for model training and evaluation (excluding canceled/diverted flights to maintain topological continuity).
