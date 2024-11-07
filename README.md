# 🧩 Toddler Autism Prediction App 🧠

The **Toddler Autism Prediction App** leverages machine learning to assess the likelihood of Autism Spectrum Disorder (ASD) in toddlers. Designed to be user-friendly for parents and caregivers, the app features an intuitive interface, streamlined data input, and robust privacy protections.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Methodology](#methodology)
- [Results](#results)
- [Contributors](#contributing)
- [License](#license)

---

## 📖 Overview

Early detection of **Autism Spectrum Disorder (ASD)** can significantly improve outcomes by providing early interventions and tailored resources for children. This app utilizes a **Naive Bayes classifier** to predict the likelihood of autism in toddlers based on behavioral and developmental data. The model is designed to be **interpretable** and **accurate**, offering caregivers insights into their child’s developmental health.

We began this project by exploring several key research papers related to **Graphical Models for Health Diagnosis**, including:

- [Employing Bayesian Networks for the Diagnosis and Prognosis of Diseases](https://arxiv.org/abs/2304.06400)
- [Bayesian Networks for the Diagnosis and Prognosis of Diseases: A Scoping Review](https://www.mdpi.com/2504-4990/6/2/58)
- [Impact of Bayesian Network Model Structure on the Accuracy of Medical Diagnostic Systems](https://ali-fahmi.github.io/files/papers/paper5.pdf)

Through this research, we gained foundational insights into Bayesian Networks, understanding their mathematical underpinnings, advantages, challenges, and applications in healthcare. A pivotal study, *Bayesian Networks in Healthcare: What is Preventing Their Adoption?* by Evangelia Kyrimi et al., informed our approach, especially in autism detection, where we found additional papers like *An Intelligent Bayesian Hybrid Approach to Help Autism Diagnosis* by Paulo Vitor de Campos Souza et al.

## ✨ Features

- **User-Friendly Interface**: Easy navigation and clear prompts for data entry.
- **Comprehensive Data Collection**: Gathers demographic info, family history, health indicators (e.g., jaundice), and behavioral assessments.
- **Behavioral Questionnaire**: Includes questions evaluating developmental behaviors like eye contact and social interaction.
- **Machine Learning Predictions**: Uses a Naive Bayes classifier to analyze user input and predict ASD likelihood.
- **Clear Results and Insights**: Provides probability-based predictions with interpretative insights.
- **Educational Resources**: Links to information on autism signs and early detection importance.

## 🛠️ Prerequisites

- **Python 3.x** installed on your machine and it's various relevant libraries such as pip.
- Knowledge of MERN, Flask


## 🧬 Methodology

1. **Data Collection and Preparation**: We used an autism screening dataset from [Kaggle](https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers). We cleaned and preprocessed the data, addressing missing values and normalizing attributes.
2. **Modeling with Naive Bayes**: Our initial classifier was Naive Bayes, chosen for its simplicity and effectiveness in handling probabilistic predictions. We also experimented with **Random Forest** and **Ensemble Models** for comparison.
3. **Frontend Implementation**: To showcase the Naive Bayes classifier, we built a basic frontend using **HTML, CSS, and Vanilla JS**.
4. **Testing and Evaluation**: We compared model accuracy across Naive Bayes, Random Forest, and Ensemble Models to determine the most reliable predictor. 
5. The whole project can be viewed on this repository.

## 📈 Results

The app provides **probabilistic predictions** with an emphasis on **interpretability**, helping caregivers understand which factors significantly influence autism likelihood.

## Additional References and Resources

We referred to a wide range of resources throughout the project, including:
- [QCHAT-10 Autism Survey for Toddlers](https://www.autismalert.org/uploads/PDF/SCREENING--AUTISM--QCHAT-10%20Question%20Autism%20Survey%20for%20Toddlers.pdf)
- [Kaggle Learning Modules](https://www.kaggle.com/learn)
- [Machine Learning Tutorial on YouTube](https://www.youtube.com/watch?v=i_LwzRVP7bg)
- [ASD Tests Online Resource](https://www.asdtests.com/)

## 👥 Contributors
- **Aninda Paul** - Roll No: **202211001**
- **Ayush Kumar** - Roll No: **202211008**
- **Devrikh Jatav** - Roll No: **202211018**
- **Inarat Hussain** - Roll No: **202211030**

## 📜 License

This project is licensed under the **Apache License**. See the [LICENSE](LICENSE) file for details.

--- 
