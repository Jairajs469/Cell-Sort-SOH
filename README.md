# Cell-Sort-SOH
# CellSort ML: Multi-Slot 18650 Li-ion Cell Grading Station with Embedded TinyML-Based SoH Estimation

CellSort ML is an open-source, edge-intelligence battery diagnostics device designed to rapidly estimate the State of Health (SoH) of cylindrical 18650 Lithium-ion cells. By implementing a 3-to-5-minute time-domain pulse test, the system bypasses traditional multi-hour galvanostatic charge-discharge cycles to classify cell degradation profiles in real time.

---

## ⚡ Core System Features

* **On-Chip Inference Engine:** Processes full feedforward operations locally on an ESP32 microcontroller in under 1.0 ms.
* **Noise-Isolated Analog Frontend:** Uses an external 16-bit ADS1115 Delta-Sigma converter to eliminate the non-linear noise profiles of native microcontrollers.
* **Dynamic Pulse Gating:** Administers controlled micro-discharges using a logic-level IRLZ44N N-Channel MOSFET.
* **Real-Time Visual Dashboard:** Automatically prints live OCV, transient internal resistance, and predicted health grades to an SSD1306 OLED panel.
* **Strict Cost Optimization:** Designed, fabricated, and fully hand-soldered on an FR4 single-sided dot-matrix perfboard under a rigid ₹1,500 budget boundary.

---

## 🧠 TinyML Modeling & Performance Metrics

The embedded pipeline executes the **CellSort V2 Unweighted Invariant Features** Multi-Layer Perceptron (MLP) neural network natively on-chip[cite: 1]. 

* **Baseline Training Dataset:** Trained and validated using the state-of-the-art **KIT Dataset (Karlsruhe Institute of Technology)**[cite: 1].
* **Mean Absolute Error (MAE):** **2.996%** on held-out validation sequences[cite: 1].
* **Coefficient of Determination ($R^2$):** **0.9138**, demonstrating high capture of electro-chemical variance[cite: 1].
* **Invariant Feature Matrix:** Utilizes **6 unique time-domain parameters** engineered specifically to be the least C-rate dependent features[cite: 1]. This prevents current fluctuations from degrading prediction precision under active load loops.

### Permutation Feature Importance Rankings
Features are ranked by the mean performance drop (increase in MAE %) observed when inputs are shuffled[cite: 1]:
1. `IR_10ms_mOhm` (**7.3%** drop): Dominant transient internal resistance under initial step-load[cite: 1].
2. `IR_norm` (**6.1%** drop): Normalized internal resistance mapping state deviation[cite: 1].
3. `IR_1s_mOhm` (**5.2%** drop): Steady DC internal resistance at the 1-second pulse plateau[cite: 1].
4. `IR_ratio` (**4.8%** drop): Ratiometric tracking of transient vs. steady resistance[cite: 1].
5. `ocv_est_end_V` (**3.8%** drop): Extrapolated Open Circuit Voltage relaxation asymptote[cite: 1].
6. `ocv_norm` (**2.6%** drop): Normalized baseline initial state voltage[cite: 1].

---

## 📁 Repository Directory Layout

```text
cellsort-ml/
├── data_and_model/
│   ├── cellsort_ml_consolidated_data.txt    # Empirical pipeline data reports
│   └── model.tflite                         # Binary network file for Netron visualization
├── firmware/
│   ├── firmware.ino                         # Main embedded control loop state machine
│   └── model.h                              # C++ quantized byte array header wrapper
└── README.md                                # Repository landing dossier
