# Scalable Management of UAV Traffic in Dense Urban Low-Altitude Airspace

This repository provides the data, code, and supplementary materials for the paper:

> **Scalable management of unmanned aerial vehicle traffic in dense urban low-altitude airspace**  
> Qingying He, Wei Liu*, Hai Yang  

## 📌 Overview

Urban air mobility is rapidly transitioning from pilot deployments to large-scale systems. A central challenge is how to allocate limited low-altitude airspace among a large number of competing UAV operations in real time.

This project develops a **market-based airspace allocation framework** based on:

- A **spatiotemporal airspace network representation**
- A **combinatorial auction model** for route allocation
- A **pricing-based primal–dual online algorithm**

##  📊 Data
This repository includes relevant datasets to support reproducibility.

### 1. Platform-specific datasets

The following real-world datasets are used in this study and can be accessed via official sources:

- **Meituan delivery dataset**  
  - ~568,545 on-demand delivery records  
  - Used for urban logistics analysis  
  - Access: [https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge]

- **Cainiao parcel delivery dataset**  
  - >7.5 million delivery records across multiple cities  
  - Used for first- and last-mile evaluation  
  - Access: [https://huggingface.co/datasets/Cainiao-AI/LaDe-P] and [https://huggingface.co/datasets/Cainiao-AI/LaDe-D]

> These datasets are subject to platform-specific data policies. Users should comply with the corresponding terms of use when accessing and using the data.

---

### 2. Repository datasets

The following datasets are used in this study. Due to file size limitations, some datasets are hosted externally via Google Drive.

- **Shenzhen locations dataset**
  - ~5,000 anonymized locations sampled from Shenzhen, China  
  - Provided in the repository (`data/`)

- **Processed Meituan dataset**
  - Based on 568,545 on-demand delivery trips with anonymized pickup–dropoff locations and timestamps  
  - Processed to extract origin–destination pairs and feasible time windows, and to construct UAV counterparts using Euclidean distance (45 km/h)  
  - Restricted to high-demand locations, defined as the top 20% most frequent pickup–delivery pairs  
  - Used in experiments in Section 2.3  
  - Provided in the repository (`data/`)

- **Processed Cainiao dataset**
  - Derived from large-scale parcel delivery data  
  - Includes structured O–D pairs for first- and last-mile analysis  
  - Used for cross-city evaluation (Section 2.3)  
  - Access (processed data): [https://drive.google.com/file/d/1bnSXyNDqzhmksw_w7pEcSpb1QKeil7_s/view?usp=sharing] and [https://drive.google.com/file/d/1Bq5_a4Wtyaz7lU0CPxBzOCUUOq8wLn6y/view?usp=sharing]

- **Amap POI dataset**
  - Nationwide point-of-interest data  
  - Used in Section 2.4  
  - Access: [https://drive.google.com/file/d/1kKIdRMjxHK_ZIo-6fMMBa_h6EPyb0eL0/view?usp=sharing]


---

### 3. Data availability statement

All synthetic and reproducible datasets are included in this repository  

This setup ensures compliance with data usage policies while maintaining reproducibility of the main results.


## 🧠 Code

The code is organized under the `code/` directory and supports simulation and experiment reproduction.

### 1. Simulation

- `code/simulation/`
  - Implements the airspace allocation simulation  
  - Includes demand generation, and online allocation  
  - Corresponds to numerical experiments in Section 2.2  

---

### 2. Experiments

- `code/experiments/`
  - Reproduces numerical results in the paper  
  - Includes empirical analysis using Meituan and Cainiao data (Section 2.3)  
