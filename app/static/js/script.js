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
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Received data:", data);  // Debug log
        
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
        
        // Show explanation section
        const explanationSection = document.getElementById("explanation");
        explanationSection.style.display = 'block';
        
        // Reset all sections
        document.getElementById("phishingIndicators").style.display = 'none';
        document.getElementById("legitimateIndicators").style.display = 'none';
        document.getElementById("phishingIndicatorsList").innerHTML = '';
        document.getElementById("legitimateIndicatorsList").innerHTML = '';
        document.getElementById("recommendationText").textContent = '';
        
        if (data.explanation) {
            const lines = data.explanation.split('\n');
            let currentSection = null;
            
            lines.forEach(line => {
                if (line.includes("This email has been classified")) {
                    document.getElementById("explanationText").textContent = line;
                } else if (line.includes("Potential phishing indicators:")) {
                    currentSection = "phishing";
                    document.getElementById("phishingIndicators").style.display = 'block';
                } else if (line.includes("Factors suggesting legitimacy:")) {
                    currentSection = "legitimate";
                    document.getElementById("legitimateIndicators").style.display = 'block';
                } else if (line.includes("Recommendation:")) {
                    currentSection = "recommendation";
                } else if (line.trim() && line.startsWith("-")) {
                    const listItem = document.createElement("li");
                    listItem.textContent = line.substring(1).trim();
                    
                    if (currentSection === "phishing") {
                        document.getElementById("phishingIndicatorsList").appendChild(listItem);
                    } else if (currentSection === "legitimate") {
                        document.getElementById("legitimateIndicatorsList").appendChild(listItem);
                    } else if (currentSection === "recommendation") {
                        document.getElementById("recommendationText").textContent += line.substring(1).trim() + " ";
                    }
                }
            });
        }
        
        // VirusTotal Result
        if (data.virustotal) {
            document.getElementById("virustotalResult").textContent = data.virustotal;
        }
        
        // Show Learn More button
        document.getElementById("learnMore").style.display = 'block';
    })
    .catch(error => {
        console.error('Detailed error:', error);  
        document.getElementById("resultText").textContent = `Error: ${error.message}`;
        document.getElementById("explanation").style.display = 'none';
    });
});

document.getElementById("clearButton").addEventListener("click", function() {
    document.getElementById("inputContent").value = '';
    document.getElementById("detectionResult").style.display = 'none';
    document.getElementById("explanation").style.display = 'none';
    document.getElementById("teachingMoment").style.display = 'none';
    document.getElementById("learnMore").style.display = 'none';
    document.getElementById("virustotalResult").textContent = '';
});