async function predict() {
    try {
        
        const form = document.getElementById('autismForm');
        const resultDiv = document.getElementById('result');
      
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

    
        const formData = {
            sex: document.getElementById('sex').value,
            ethnicity: document.getElementById('ethnicity').value,
            jaundice: document.getElementById('jaundice').value,
            family_with_asd: document.getElementById('family_with_asd').value,
            who_completed_test: document.getElementById('who_completed_test').value,
            age_mons: document.getElementById('age_mons').value
        };

  
        for (let i = 1; i <= 10; i++) {
            const scoreKey = `a${i}_score`;
            const selectedRadio = document.querySelector(`input[name="${scoreKey}"]:checked`);
            if (selectedRadio) {
                formData[scoreKey] = selectedRadio.value;
            }
        }

     
        resultDiv.style.display = 'block';
        resultDiv.textContent = 'Processing...';

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
      
        resultDiv.style.display = 'block';
        resultDiv.textContent = `Prediction: ${result.prediction}%`;
        resultDiv.style.backgroundColor = '#d4edda';

    } catch (error) {
        console.error('Error:', error);
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'block';
        resultDiv.textContent = 'An error occurred. Please try again.';
        resultDiv.style.backgroundColor = '#f8d7da';
    }
}