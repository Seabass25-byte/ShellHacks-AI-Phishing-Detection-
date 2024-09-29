document.getElementById("detectionForm").addEventListener("submit", function(event) {
    event.preventDefault();
    
    let content = document.getElementById("inputContent").value;
    if (content.trim() === "") {
        alert("Please enter a URL or email.");
        return;
    }

    // Show loading text
    document.getElementById("resultText").textContent = "Analyzing... Please wait.";
    document.getElementById("detectionResult").style.display = 'block';

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: content })
    })
    .then(response => response.json())
    .then(data => {
        // Set the color based on the result
        const resultBox = document.getElementById("detectionResult");
        if (data.final_result === "phishing") {
            resultBox.classList.add("result-danger");
            resultBox.classList.remove("result-success");
        } else {
            resultBox.classList.add("result-success");
            resultBox.classList.remove("result-danger");
        }

        // Display result and confidence
        document.getElementById("resultText").textContent = `Result: ${data.final_result.charAt(0).toUpperCase() + data.final_result.slice(1)} Detected`;
        document.getElementById("confidence").textContent = `Confidence: ${data.phishing_confidence.toFixed(2)}%`;

        // Explanation
        document.getElementById("explanation").textContent = data.explanation;

        // Teaching Tip
        if (data.tips && data.tips.length > 0) {
            document.getElementById("teachingMoment").style.display = 'block';
            document.getElementById("teachingTip").textContent = data.tips.join(" ");
        }

        // Show Learn More button
        document.getElementById("learnMore").style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("resultText").textContent = "An error occurred. Please try again.";
    });
});

document.getElementById("clearButton").addEventListener("click", function() {
    document.getElementById("inputContent").value = '';
    document.getElementById("detectionResult").style.display = 'none';
    document.getElementById("teachingMoment").style.display = 'none';
    document.getElementById("learnMore").style.display = 'none';
});
