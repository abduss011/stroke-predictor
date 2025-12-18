document.getElementById('predictForm').onsubmit = async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target));

    data.age = parseInt(data.age);
    data.hypertension = parseInt(data.hypertension);
    data.heart_disease = parseInt(data.heart_disease);
    data.avg_glucose_level = parseFloat(data.avg_glucose_level);
    data.bmi = parseFloat(data.bmi);

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (!response.ok) {
            alert('Server Error: ' + (result.error || 'Unknown error'));
            return;
        }

        document.getElementById('riskLevel').innerText = result.risk_level;
        document.getElementById('probability').innerText = (result.stroke_probability * 100).toFixed(2) + '%';
        document.getElementById('recText').innerText = result.recommendation;

        resultDiv.className = result.risk_level.toLowerCase();
        resultDiv.style.display = 'block';
    } catch (err) {
        alert('Prediction failed. Is the server running?');
    }
};