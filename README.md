# sEMGxRoboticHand

Recent advances in **surface electromyography (sEMG) decoding**, such as **[Meta’s EMG2Pose](https://arxiv.org/abs/2412.02725)**, **[EMG2QWERTY datasets](https://arxiv.org/abs/2410.20081)**, and their associated pretrained models, have demonstrated high-accuracy hand-pose and typing reconstruction. However, these breakthroughs rely on **Meta’s proprietary acquisition hardware (sEMG-RD)**, limiting reproducibility and broader utility for independent research and open development.



## Hand Pose & Reconstruction Examples
<p align="center">
  <img src="images/hand_reconstruction.png" width="85%">
</p>



## EMG-RD Wristband & Interaction Setup
<p align="center">
  <img src="images/emg_rd_setup.png" width="75%">
</p>



## Project Description

To address this gap, we present **open-sEMG-16**, a fully **open-source**, **16-channel**, **wrist-wearable sEMG acquisition system** designed with specifications roughly matched to Meta’s proprietary platform, including a **4 kHz sampling rate** and a high-fidelity 24-bit analog front-end.
The goal of this project is to replicate Meta’s sEMG-RD architecture using low-cost, commercially available components, and to evaluate whether comparable or superior performance can be achieved through optimized analog design and modular firmware.



## System Architecture

The system integrates dual ADS1298 24-bit ADCs for synchronized multi-channel acquisition, an ESP32-S3 microcontroller for real-time Wi-Fi/BLE streaming, and dry gold-plated pogo-pin electrodes arranged circumferentially around the wrist.

By maintaining compatibility with vEMG2Pose and similar models, open-sEMG-16 enables direct benchmarking and reproducible validation against Meta’s datasets, serving as a practical platform for open, repeatable sEMG research.



## Contributors
- Emir Sahin
- Lia Brahami
- Katherine Lambert
- Karen Chen Lai



##  References

**[1] EMG2Pose — Meta AI Research (2024)**
https://arxiv.org/abs/2410.20081

**[2] EMG2QWERTY — Meta AI Research (2024)**
https://arxiv.org/abs/2410.20081

**[3] A generic non-invasive neuromotor interface for human-computer interaction (2025)**
https://www.nature.com/articles/s41586-025-09255-w

**[4] Advancing Neuromotor Interfaces by Open Sourcing Surface Electromyography (sEMG) Datasets for Pose Estimation and Surface Typing (2024)**
https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/