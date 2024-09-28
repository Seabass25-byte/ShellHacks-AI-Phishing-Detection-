document.getElementById("detectionForm").addEventListener("submit", function(event) {
    event.preventDefault();
    let content = document.getElementById("inputContent").value;

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: content })
    })
    .then(response => response.json())
    .then(data => {
        let resultText = "Detection Result:\n";

        if (data.url_results) {
            data.url_results.forEach(urlResult => {
                resultText += `URL: ${urlResult.url}, Phishing Confidence: ${urlResult.phishing_confidence.toFixed(2)}%, Legitimate Confidence: ${urlResult.legitimate_confidence.toFixed(2)}%\n`;
            });
        }

        if (data.email_result) {
            resultText += `Email Result: ${data.email_result}\n`;
            resultText += `Phishing Confidence: ${data.phishing_confidence.toFixed(2)}%\n`;
            resultText += `Legitimate Confidence: ${data.legitimate_confidence.toFixed(2)}%\n`;

            // Educational information
            let educationText = `Education:\n`;
            if (data.email_result === 'phishing') {
                educationText += "Phishing emails often use urgent language, suspicious links, or fake sender addresses to trick users. Review this email for these signs.";
            } else {
                educationText += "This email seems legitimate. Always double-check for unexpected links or requests for personal information.";
            }

            document.getElementById("educationResult").textContent = educationText;
        }

        document.getElementById("detectionResult").textContent = resultText;
    })
    .catch(error => console.error('Error:', error));
});
