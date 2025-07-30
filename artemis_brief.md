# ARTEMIS Brief

**Student:** Steve Ren
**Date:** 2025-07-29

## Summary of ARTEMIS Method

ARTEMIS is a Graph Neural Network (GNN) system designed to detect "airdrop hunters" in NFT markets. Airdrop hunters are users who create numerous fake accounts (Sybil attacks) to unfairly acquire airdropped tokens, undermining the goal of building genuine, decentralized communities.

The core idea of ARTEMIS is to represent the NFT market as a complex graph where wallet addresses are nodes and transactions are edges. It then uses a specialized GNN to learn the behavioral patterns that distinguish hunters from legitimate users. The method's strength comes from three key innovations:

1.  **Multimodal Feature Engineering:** ARTEMIS goes beyond simple transaction data. It analyzes the NFTs themselves by using Transformer models (like ViT and BERT) to extract features from their images and text metadata. This provides rich context about *what* is being traded.

2.  **Behavioral Feature Engineering:** The model incorporates features designed to capture suspicious trading activity based on market manipulation theories. This includes looking for patterns like wash trading, which are strong indicators of non-genuine behavior.

3.  **Custom GNN Architecture:** ARTEMIS uses a custom GNN that includes a specialized first-layer convolution (`ArtemisFirstLayerConv`) to effectively combine the node and edge features. It also employs a tailored neighborhood sampler (`NeighborSamplerbyNFT`) to focus the learning process on the most informative transaction sequences for identifying hunters.

By combining these three elements, ARTEMIS creates a rich, multi-faceted representation of each user's behavior, allowing it to achieve high accuracy in identifying airdrop hunters. The official code was successfully run, and the model training converged as expected, achieving a final F1 Score of **0.964** on the test set in the first run, confirming the results reported in the paper.

## Observed Limitation or Weakness

A concrete limitation of the ARTEMIS method is its **reliance on a static, post-hoc dataset**. The system is trained and evaluated on a snapshot of historical transaction data. While it is effective at identifying hunters who have already performed their activities, it has two related weaknesses:

1.  **Not Real-Time:** It is not designed for real-time prevention. By the time the analysis is run and hunters are identified, they have already claimed the airdropped assets. The damage to the project's launch fairness has already been done. A project would have to use this tool to retroactively penalize accounts or try to claw back assets, which is often difficult or impossible.

2.  **Vulnerability to Adversarial Drift:** The features engineered by ARTEMIS are based on the *known* behaviors of past airdrop hunters. Sophisticated hunters could analyze this very paper (or similar research) and adapt their strategies to evade detection. For example, they could add random delays between transactions, use more complex trading paths to obscure wash trading, or use AI to generate more "genuine-looking" NFT metadata. Because the model is trained on a static dataset, it cannot adapt to these evolving adversarial strategies without being completely retrained on a new, labeled dataset, which is a time-consuming and expensive process.