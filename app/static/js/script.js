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
        resultText += `Phishing Confidence: ${data.phishing_confidence.toFixed(2)}%\n`;
        resultText += `Legitimate Confidence: ${data.legitimate_confidence.toFixed(2)}%\n`;
        resultText += `Final Result: ${data.final_result}\n\n`;
        resultText += `Explanation: ${data.explanation || 'No explanation available.'}`;
        
        document.getElementById("detectionResult").textContent = resultText;
    })
    .catch(error => console.error('Error:', error));
});
