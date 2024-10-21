
        async function predict() {
            const data = {
                Sex: document.getElementById('sex').value,
                Ethnicity: document.getElementById('ethnicity').value,
                Jaundice: document.getElementById('jaundice').value,
                Family_mem_with_ASD: document.getElementById('family_with_asd').value,
                "Who completed the test": document.getElementById('who_completed_test').value,
                A1_Score: document.querySelector('input[name="a1_score"]:checked').value,
                A2_Score: document.querySelector('input[name="a2_score"]:checked').value,
                A3_Score: document.querySelector('input[name="a3_score"]:checked').value,
                A4_Score: document.querySelector('input[name="a4_score"]:checked').value,
                A5_Score: document.querySelector('input[name="a5_score"]:checked').value,
                A6_Score: document.querySelector('input[name="a6_score"]:checked').value,
                A7_Score: document.querySelector('input[name="a7_score"]:checked').value,
                A8_Score: document.querySelector('input[name="a8_score"]:checked').value,
                A9_Score: document.querySelector('input[name="a9_score"]:checked').value,
                A10_Score: document.querySelector('input[name="a10_score"]:checked').value,
                Age_Mons: document.getElementById('age_mons').value
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            displayResult(result);
        }

        function displayResult(result) {
            const resultContainer = document.getElementById('result');
            resultContainer.innerHTML = '';

            for (const key in result) {
                if (result.hasOwnProperty(key)) {
                    const percentage = (result[key] * 100).toFixed(2); 
                    const resultDiv = document.createElement('div');
                    resultDiv.classList.add('result');
                    resultDiv.innerHTML = `<span class="percentage">${key}: ${percentage}%</span>`;
                    resultContainer.appendChild(resultDiv);
                }
            }
        }
