# 🧩 Toddler Autism Prediction App 🧠

The **Toddler Autism Prediction App** leverages machine learning to assess the likelihood of Autism Spectrum Disorder (ASD) in toddlers. Designed to be user-friendly for parents and caregivers, the app features an intuitive interface, streamlined data input, and robust privacy protections.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Methodology](#methodology)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)
  

---

## Overview

Early detection of **Autism Spectrum Disorder (ASD)** can significantly improve outcomes by providing early interventions and tailored resources for children. This app utilizes a **Naive Bayes classifier** to predict the likelihood of autism in toddlers based on behavioral and developmental data. The model is designed to be **interpretable** and **accurate**, offering caregivers insights into their child’s developmental health.

We began this project by exploring several key research papers related to **Graphical Models for Health Diagnosis**, including:

- [Detection of Autism Spectrum Disorder in Children Using Machine Learning Techniques](https://link.springer.com/article/10.1007/s42979-021-00776-5)
- [An accessible and efficient autism screening method for behavioural data and predictive analyses](https://journals.sagepub.com/doi/10.1177/1460458218796636)
- [Employing Bayesian Networks for the Diagnosis and Prognosis of Diseases](https://arxiv.org/abs/2304.06400)
- [Bayesian Networks for the Diagnosis and Prognosis of Diseases: A Scoping Review](https://www.mdpi.com/2504-4990/6/2/58)
- [An Intelligent Bayesian Hybrid Approach to Help Autism Diagnosis* by Paulo Vitor de Campos Souza et al](https://link.springer.com/article/10.1007/s00500-021-05877-0)
- [Impact of Bayesian Network Model Structure on the Accuracy of Medical Diagnostic Systems](https://ali-fahmi.github.io/files/papers/paper5.pdf)

Through this research, we gained foundational insights into Bayesian Networks, understanding their mathematical underpinnings, advantages, challenges, and applications in healthcare.One of the pivotal study, *[Detection of Autism Spectrum Disorder in Children Using Machine Learning Techniques](https://link.springer.com/article/10.1007/s42979-021-00776-5)*.

## Features

- **User-Friendly Interface**:
  - Easy navigation and clear prompts for data entry.
- **Comprehensive Data Collection**:
  - Gathers demographic info, family history, health indicators (e.g., jaundice), and behavioral assessments.
- **Behavioral Questionnaire**:
  - Includes questions evaluating developmental behaviors like eye contact and social interaction.
- **Machine Learning Predictions**:
  - Uses a Naive Bayes classifier to analyze user input and predict ASD likelihood.
- **Clear Results and Insights**:
  - Provides probability-based predictions with interpretative insights.
- **Educational Resources**:
  - Links to information on autism signs and early detection importance.

## Prerequisites

- **Python 3.x** installed on your machine.
- Knowledge of MERN, Flask.
- Basic knowledge of terminal commands.

## Installation

 1. ### **Clone the repository**:

    ```bash
    git clone https://github.com/ayushkumar912/Toddler_Autism_Prediction.git
    cd Toddler_Autism_Prediction
    ```

 2. ### **Create a virtual environment**:

    - macOS/Linux

    ```bash
    python3 -m venv .venv
    ```

    - Windows

    ```bash
    python -m venv .venv
    ```

 3. ### **Activate the virtual environment**:

     - macOS/Linux

    ```bash
    source .venv/bin/activate 
    ```

    - Windows

    ```bash
    .venv\Scripts\activate 
    ```

 4. ### **Install the required modules**:


    ```bash
    pip install -r modules.txt
    ```

 ## Usage

 To run the project, use the following command:
 - macOS/Linux
   ```bash
    python3 app.py
   ```
 - Windows
   ```bash
     python app.py
    ```
   
 ## Build and Run Docker Container
 1. ### **Build the Docker image**:
    ```bash
    docker build -t flask-app .
    ```


 2. ### **Run the container**:
    ```bash
       docker run -p 80:80 flask-app
    ```
  

  ## Methodology

1. **Data Collection and Preparation**:
    - We used an autism screening dataset from [Kaggle](https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers). We cleaned and     preprocessed the data, addressing missing values and normalizing attributes.
3. **Modeling with Naive Bayes**:
    - Our initial classifier was Naive Bayes, chosen for its simplicity and effectiveness in handling probabilistic predictions. We also experimented     with **Random Forest** and **Ensemble Models** for comparison.
4. **Frontend Implementation**:
    - To showcase the Naive Bayes classifier, we built a basic frontend using **HTML, CSS, and Vanilla JS**.
5. **Testing and Evaluation**:
    - We compared model accuracy across Naive Bayes, Random Forest, and Ensemble Models to determine the most reliable predictor. 
6. The whole project can be viewed on this repository.

## Results

The app provides **probabilistic predictions** with an emphasis on **interpretability**, helping caregivers understand which factors significantly influence autism likelihood.

## Additional References and Resources

We referred to a wide range of resources throughout the project, including:
- [QCHAT-10 Autism Survey for Toddlers](https://www.autismalert.org/uploads/PDF/SCREENING--AUTISM--QCHAT-10%20Question%20Autism%20Survey%20for%20Toddlers.pdf)
- [Kaggle Learning Modules](https://www.kaggle.com/learn)
- [Machine Learning Tutorial on YouTube](https://www.youtube.com/watch?v=i_LwzRVP7bg)
- [ASD Tests Online Resource](https://www.asdtests.com/)

##  Contributors
- **Aninda Paul**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Roll No: **202211001**
- **Ayush Kumar**&nbsp;&nbsp;&nbsp;&nbsp;Roll No: **202211008**
- **Devrikh Jatav**&nbsp;&nbsp;&nbsp;Roll No: **202211018**
- **Inarat Hussain**&nbsp;&nbsp;Roll No: **202211030**

## License

This project is licensed under the **Apache License**. See the [LICENSE](LICENSE) file for details.

--- 
