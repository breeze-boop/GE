## LatGE
This repository provides the experimental artifact for our paper. It contains a full of implementation and evaluation of our scheme latge, and reproduces the previous fdge scheme.
 
## Code Organization
The fdge codebase includes cryptographic algorithms and modules, benchmarking utilities. These components support membership management, encryption/decryption/opening, zero-knowledge proof generation /verification and runtime evaluation. For zkp runtime, we take a cost estimation model due to too large matrix. 

The latge codebase is divided into three parts: cryptographic core, the complete scheme implementation, and lightweight helpers for orchestration and testing. It separates the reusable components from the end-to-end experimental workflow.