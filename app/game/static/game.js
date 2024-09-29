let images = []; // Array to hold image data
let currentIndex = 0; // Track the current image index

// Fetch all images when the page loads
async function loadImages() {
    const response = await fetch('/get_all_images');
    images = await response.json();
    loadImage(); // Load the first image
}

// Load the current image
function loadImage() {
    if (images.length > 0) {
        document.getElementById('gameImage').src = images[currentIndex].url;
        document.getElementById('result').innerText = ""; // Clear previous result
    }
}

// Submit user guess (phishing or not phishing)
async function submitGuess(guess) {
    const response = await fetch('/submit_guess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ guess: guess, is_phishing: images[currentIndex].is_phishing })
    });

    const resultData = await response.json();
    document.getElementById('result').innerText = `You are ${resultData.result}!`;

    // Update the current index and load the next image after 2 seconds
    currentIndex = (currentIndex + 1) % images.length; // Loop back to start
    setTimeout(loadImage, 2000); // Load next image after 2 seconds
}

window.onload = loadImages;
