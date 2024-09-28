document.getElementById("detectionForm").addEventListener("submit", function(event) {
    event.preventDefault();
    let content = document.getElementById("inputContent").value;

    // Send content to the backend for phishing detection
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ content: content })
    })
    .then(response => response.json())
    .then(data => {
        let resultText = `<h3>Detection Result:</h3>`;

        // If URLs are found, display their results
        if (data.url_results) {
            resultText += `<h4>URL Results:</h4>`;
            data.url_results.forEach(urlResult => {
                resultText += `<strong>URL:</strong> ${urlResult.url}<br>`;
                resultText += `<strong>Phishing Confidence:</strong> ${urlResult.phishing_confidence.toFixed(2)}%<br>`;
                resultText += `<strong>Legitimate Confidence:</strong> ${urlResult.legitimate_confidence.toFixed(2)}%<br><br>`;
            });
        }

        // If email content was processed, display email results
        if (data.email_result) {
            resultText += `<h4>Email Results:</h4>`;
            resultText += `<strong>Result:</strong> ${data.email_result}<br>`;
            resultText += `<strong>Phishing Confidence:</strong> ${data.phishing_confidence.toFixed(2)}%<br>`;
            resultText += `<strong>Legitimate Confidence:</strong> ${data.legitimate_confidence.toFixed(2)}%<br><br>`;
        }

        // Update the result area with formatted result text
        document.getElementById("detectionResult").innerHTML = resultText;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
