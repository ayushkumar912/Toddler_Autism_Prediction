# Toddler Autism Prediction App

The Toddler Autism Prediction App leverages machine learning to provide quick assessments of the likelihood of autism spectrum disorder (ASD) in toddlers. Designed for parents and caregivers, the app features an intuitive interface that simplifies data input and enhances user experience.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.x installed on your machine.
- Basic knowledge of terminal commands.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ayushkumar912/Toddler_Autism_Prediction.git
   cd Toddler_Autism_Prediction
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv .venv
   <!-- for Windows -->
   python -m venv .venv
   ```

3. **Activate the virtual environment**:

   ```bash
   source .venv/bin/activate 
   <!-- for Windows -->
   .venv\Scripts\activate 
   ```

4. **Install the required modules**:

   ```bash
   pip install -r modules.txt
   ```

## Usage

To run the project, use the following command:

```bash
python3 app.py
<!-- for Windows -->
python app.py
```

## Features

- **User-Friendly Interface**: Easy-to-navigate design ensures seamless data entry.

- **Comprehensive Data Collection**: Users input demographic information, health indicators (e.g., jaundice), family history of ASD, and behavioral assessments through structured questionnaires.

- **Behavioral Questionnaire**: Evaluates key developmental behaviors, such as eye contact and social interaction, to provide a holistic view of the child's progress.

- **Machine Learning Predictions**: Uses a Naive Bayes model trained on historical data to analyze inputs and predict the likelihood of ASD, offering immediate feedback on results.

- **Clear Results and Insights**: Displays the predicted likelihood of autism along with probability scores to aid interpretation.

- **Error Handling**: Ensures all required fields are completed accurately, providing users with informative prompts for corrections.

- **Privacy and Security**: Protects user data with a focus on privacy, ensuring no personal information is stored beyond the prediction process.

- **Responsive Design**: Compatible with various devices for a consistent experience on smartphones, tablets, and desktops.

- **Educational Resources**: Offers links to valuable information about autism signs and the importance of early detection.

**Conclusion**: The Toddler Autism Prediction App is a powerful tool for early autism detection, empowering caregivers to gain insights into their child's developmental health while promoting timely intervention and awareness.

## Contributing

If you want to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
