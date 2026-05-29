# Domain-Similarity Aware Federated Proximal (DSA-FedProx) for Robust Ischemic Stroke Classification in Multi-Center DWI-MRI

Official repository of the paper: **"Federated Learning for Robust Ischemic Stroke Classification in Multi-Center DWI"** [17].

Developed by the **Machine Learning Analysis and Computer Vision (M_L_ACV)** and the **Biomedical Imaging, Vision and Learning Laboratory (BIVL²ab)** at Universidad Industrial de Santander (UIS), Bucaramanga, Colombia.

---

## 👥 Authors
* **Nicolás Linares Rojas** - [nicolas.linares@correo.uis.edu.co](mailto:nicolas.linares@correo.uis.edu.co)
* **Isabel Sofía García Joya** 
* **Alejandra Díaz Navarro**

---

## 🌟 Key Highlights
* **Outstanding Performance:** Achieved a **ROC-AUC of 0.94**, **Recall (Sensitivity) of 0.88**, and **Specificity of 0.80** on a completely unseen, blind test clinical center (Center 5), overcoming severe out-of-distribution (OOD) domain shifts.
* **DSA-FedProx Aggregation:** A novel hybrid federated aggregation rule that balances database volume with target-domain clinical compatibility using Wasserstein distance metrics [15, 42].
* **Clinical Feature Fusion:** Implements a custom 3-channel input configuration `[DWI, ADC, DWI - ADC]` paired with localized slice-level Z-score normalization to eliminate scanner scale discrepancies [3, 4].
* **Test-Time Augmentation (TTA):** Employs spatial transformations during inference to reduce prediction variance on unseen domain centers [26].

---

## 📁 Repository Structure
```text
BIVL2ab_Stroke_Federated/
├── notebooks/
│   └── Stroke_Federated_Inference.ipynb   # Complete Google Colab pipeline
├── report/
│   └── Ischemic_Stroke_Preprint.pdf       # Under evaluation preprint PDF
└── README.md                              # This documentation file
```

---

## 🧠 Proposed Methodology: DSA-FedProx

Our framework attacks the clinical challenge of multi-center intensity covariate shifts at both the local preprocessing and global aggregation levels.

### 1. Local Preprocessing & Feature Engineering
Medical MRI scanners operate on variable intensity scales. To neutralize this, we apply an independent slice-level Z-score normalization computed exclusively over the brain mask tissue. To simulate clinical decision-making, we stack the inputs into a 3-channel tensor `[DWI, ADC, DWI-ADC]`, where the third channel explicitly amplifies restricted water diffusion.

### 2. Domain-Similarity Federated Aggregation
We propose the **DSA-FedProx** rule, which computes a hybrid aggregation weight $\kappa_k$ governed by a convex combination parameter $\alpha \in [0, 1]$ [15, 40]:

$$\kappa_k = \alpha \cdot \text{Prior}_{\text{Size}}(k) + (1 - \alpha) \cdot \text{Prior}_{\text{DomainSimilarity}}(k)$$

where the size prior is the standard FedAvg coefficient and the domain similarity prior is inversely proportional to the Wasserstein distance ($W$) between the training client $k$ and the target Center 5.

Local training is regularized by a proximal term to mitigate client drift across heterogeneous domains [42]:

$$\mathcal{L}_k^{\text{total}} = \mathcal{L}_k^{\text{Focal}} + \frac{\mu}{2} \|w_i - w^t\|^2$$

---

## 📊 Results & Performance Benchmark
Below is the performance benchmark evaluated on the out-of-distribution **Center 5 (Blind Test)**:

| Model & Strategy | Acc | Prec | Recall | Spec | F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. FedAvg + EfficientNet-B0 (Baseline) | 0.72 | 0.58 | 0.66 | 0.75 | 0.62 | 0.81 | 0.78 |
| 2. DSA-FedProx + DenseNet121 | 0.76 | 0.61 | 0.87 | 0.71 | 0.72 | 0.90 | 0.88 |
| 3. DSA-FedProx + EfficientNet-B0 | 0.80 | 0.69 | 0.75 | **0.82** | 0.72 | 0.88 | 0.86 |
| **4. DSA-FedProx + EfficientNet-B3 + TTA (Ours)** | **0.82** | **0.70** | **0.88** | 0.80 | **0.78** | **0.94** | **0.92** |
| 5. DSA-FedProx + DenseNet121 + TTA | 0.70 | 0.55 | 0.85 | 0.62 | 0.67 | 0.88 | 0.87 |
| 6. Ensemble (DenseNet121 + EfficientNet-B3) + TTA | 0.74 | 0.59 | 0.87 | 0.68 | 0.70 | 0.92 | 0.90 |

---

## 🔒 Data Privacy & Compliance Notice
The clinical MRI datasets (Centers 1 to 5) are highly protected under patient anonymity and privacy regulations (such as HIPAA and GDPR) [13]. Therefore, raw medical images are **not** hosted on this public repository. 

To run the pipeline locally or on Google Colab, you must:
1. Obtain permission from the **Biomedical Imaging, Vision and Learning Laboratory (BIVL²ab)** or the respective ethical committees [17].
2. Structure the dataset directory in your Google Drive or local path following this format:
   ```text
   20261_Post-MLACV/
   ├── center_1/
   │   └── train/
   │       └── png/        # Contains PXXXX_SXXXX_LABEL_MOD.png files
   ├── center_5/
   │   └── test/
   │       └── png/        # Contains unlabelled test files
   ```

---

## 💻 How to Run the Notebook
1. Open the notebook `notebooks/Stroke_Federated_Inference.ipynb` in **Google Colab**.
2. Mount your Google Drive where the dataset directory is hosted.
3. Configure the `BASE_PATH` variable in the data loading cell.
4. Execute the cells sequentially. The pipeline supports training on GPU (T4, L4, or A100 environments).

---

## 📄 Citation
If you find this work or code useful for your research, please cite our preprint:

```bibtex
@article{linares2026stroke,
  title={Federated Learning for Robust Ischemic Stroke Classification in Multi-Center DWI},
  author={Linares Rojas, Nicol{\'a}s and Garc{\'\i}a Joya, Isabel Sof{\'\i}a and D{\'\i}az Navarro, Alejandra},
  journal={Preprint submitted to BIVL2ab Research Group},
  year={2026},
  url={https://github.com/nicolaslinares/BIVL2ab_Stroke_Federated}
}
```
