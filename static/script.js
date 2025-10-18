
document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const ticker = document.getElementById('ticker').value;
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const forceRetrain = document.getElementById('force-retrain').checked;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            ticker: ticker, 
            start_date: startDate, 
            end_date: endDate,
            force_retrain: forceRetrain
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        const metricsDiv = document.getElementById('metrics');
        metricsDiv.innerHTML = `
            <p><strong>Model Performance:</strong> MSE: ${data.mse}, R-squared: ${data.r2}</p>
            <p><strong>Correlation with S&P 500:</strong> ${data.correlation}</p>
            <p><strong>Annualized Volatility:</strong> ${data.volatility}</p>
            <p><strong>Next Day's Predicted Price:</strong> ${data.next_day_prediction}</p>
        `;

        const ctx = document.getElementById('price-chart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [
                    {
                        label: 'Actual Prices',
                        data: data.actual,
                        borderColor: 'blue',
                        fill: false
                    },
                    {
                        label: 'Predicted Prices',
                        data: data.predicted,
                        borderColor: 'orange',
                        linestyle: '--',
                        fill: false
                    }
                ]
            }
        });
    });
});

document.getElementById('historical-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const ticker = document.getElementById('ticker').value;
    const historicalDate = document.getElementById('historical-date').value;

    if (!ticker) {
        alert('Please enter a stock ticker first.');
        return;
    }

    fetch('/historical-predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            ticker: ticker, 
            date: historicalDate 
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        const historicalResultDiv = document.getElementById('historical-result');
        historicalResultDiv.innerHTML = `
            <p>For date: ${historicalDate}</p>
            <p><strong>Model's Prediction:</strong> ${data.predicted_price}</p>
            <p><strong>Actual Price:</strong> ${data.actual_price}</p>
        `;
    });
});
