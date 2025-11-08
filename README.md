# sEMGxRoboticHand

Recent advances in surface electromyography (sEMG) decoding, such as Meta’s EMG2Pose and EMG2QWERTY datasets and their associated pretrained models, have demonstrated high-accuracy hand-pose and typing reconstruction. However, these breakthroughs rely on Meta’s proprietary acquisition hardware (sEMG-RD), limiting the reproducibility and broader utility of their models for independent research and open development.

---

## Project Description

To address this gap, we present open-sEMG-16, a fully open-source, 16-channel wrist-wearable sEMG acquisition system designed with specifications roughly matched to Meta’s proprietary platform, including a 2 kHz sampling rate and high-fidelity 24-bit analog front-end. The goal of this project is to replicate Meta’s sEMG-RD architecture using low-cost, commercially available components and to evaluate whether comparable or superior performance can be achieved through optimized analog design and modular firmware.

---

## System Architecture

The system integrates dual ADS1298 24-bit ADCs for synchronized multi-channel acquisition, an ESP32-S3 microcontroller for real-time Wi-Fi/BLE streaming, and dry gold-plated pogo-pin electrodes arranged circumferentially around the wrist. By maintaining compatibility with vEMG2Pose and similar models, open-sEMG-16 enables direct benchmarking and validation against Meta’s datasets, serving as a practical platform for open, reproducible sEMG research.

---
