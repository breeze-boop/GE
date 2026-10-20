## LatGE
This project is an implementation of lattice-based group encryption schemes. Moreover, it provides an experimental implementation of the scheme presented in 《Lattice-Based Group Encryptions with Full Dynamicity and Message Filtering Policy》. Runtime experiments cover group sizes from 16 to 1024 users.
 
## Code Organization
The latge codebase is divided into three parts: cryptographic core, the complete scheme implementation, and lightweight helpers for orchestration and testing. It separates the reusable components from the end-to-end experimental workflow.

The fdge codebase includes cryptographic primitives, group protocol algorithms, and benchmarking utilities. These components support lattice-based encryption, membership management, commitments, and proofs, alongside calibration, repeated measurements, and runtime estimation.

Runtime results for Init, P, and V are obtained through cost-model estimation.