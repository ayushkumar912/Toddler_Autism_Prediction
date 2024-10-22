async function predict() {
    const data = {
        Sex: document.getElementById('sex').value,
        Ethnicity: document.getElementById('ethnicity').value,
        Jaundice: document.getElementById('jaundice').value,
        Family_mem_with_ASD: document.getElementById('family_with_asd').value,
        "Who completed the test": document.getElementById('who_completed_test').value,
        A1_Score: parseInt(document.querySelector('input[name="a1_score"]:checked').value),
        A2_Score: parseInt(document.querySelector('input[name="a2_score"]:checked').value),
        A3_Score: parseInt(document.querySelector('input[name="a3_score"]:checked').value),
        A4_Score: parseInt(document.querySelector('input[name="a4_score"]:checked').value),
        A5_Score: parseInt(document.querySelector('input[name="a5_score"]:checked').value),
        A6_Score: parseInt(document.querySelector('input[name="a6_score"]:checked').value),
        A7_Score: parseInt(document.querySelector('input[name="a7_score"]:checked').value),
        A8_Score: parseInt(document.querySelector('input[name="a8_score"]:checked').value),
        A9_Score: parseInt(document.querySelector('input[name="a9_score"]:checked').value),
        A10_Score: parseInt(document.querySelector('input[name="a10_score"]:checked').value),
        Age_Mons: parseInt(document.getElementById('age_mons').value)
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResult(result);
    } catch (error) {
        console.error('Error:', error);
    }
}

function displayResult(result) {
    const resultContainer = document.getElementById('result');
    resultContainer.innerHTML = '';

    // Displaying results from the prediction
    const predictionDiv = document.createElement('div');
    predictionDiv.classList.add('result');
    predictionDiv.innerHTML = `<span class="percentage">Prediction: ${result.prediction}</span>`;
    resultContainer.appendChild(predictionDiv);

    // Display probabilities
    const probabilities = ['probability', 'gaussian_probability', 'bernoulli_probability'];
    probabilities.forEach(key => {
        const resultDiv = document.createElement('div');
        resultDiv.classList.add('result');
        resultDiv.innerHTML = `<span class="percentage">${key.replace('_', ' ').toUpperCase()}: ${result[key]}</span>`;
        resultContainer.appendChild(resultDiv);
    });
    
}
