// TODO: Query for button with an id "theme-button"
let themeButton = document.getElementById("theme-button");


// TODO: Complete the toggleDarkMode function
const toggleDarkMode = () => {
  document.body.classList.toggle("dark-mode");
}


// TODO: Register a 'click' event listener for the theme button
// Set toggleDarkMode as the callback function.
themeButton.addEventListener("click", toggleDarkMode);


//signNowButton.addEventListener('click', validateForm);

let animation = {
  revealDistance: 150,
  initialOpacity: 0,
  transitionDelay: 0,
  transitionDuration: '2s',
  transitionProperty: 'all',
  transitionTimingFunction: 'ease'
};

let revealableContainers = document.querySelectorAll('.revealable');

function reveal() {
  let windowHeight = window.innerHeight;

  for (let i = 0; i < revealableContainers.length; i++) {
    let topOfRevealableContainer = revealableContainers[i].getBoundingClientRect().top;

    if (topOfRevealableContainer < windowHeight - animation.revealDistance) {
      revealableContainers[i].classList.add('active');
    } else {
      revealableContainers[i].classList.remove('active');
    }
  }
}

window.addEventListener('scroll', reveal);

const reduceMotionButton = document.getElementById('reduce-motion-button');
reduceMotionButton.addEventListener('click', reduceMotion);

function reduceMotion() {
  // Define the new animation settings to reduce motion
  animation.transitionTimingFunction = 'ease';
  animation.revealDistance = 10;  // Adjust the reveal distance as needed
  animation.transitionDuration = '0s';  // Adjust the duration as needed

  // Loop through revealable containers and update their styles
  for (let i = 0; i < revealableContainers.length; i++) {
    revealableContainers[i].style.transitionTimingFunction = animation.transitionTimingFunction;
    revealableContainers[i].style.transform = `translateY(${animation.revealDistance}px)`;
    revealableContainers[i].style.transitionDuration = animation.transitionDuration;
  }
}



// Query for the sign now button and assign it to the variable signNowButton
const signNowButton = document.getElementById("signNowButton");

// Initialize the count variable with the starting number of signatures
let count = 3;

// Function to add a signature to the webpage
const validateForm = (event) => {
  event.preventDefault(); // Prevent the default form submission
  let containsErrors = false;
  let petitionInputs = document.getElementById("sign-petition").elements;

  let person = {
    name: petitionInputs[0].value,
    hometown: petitionInputs[1].value,
    email: petitionInputs[2].value
  };

  for (let i = 0; i < petitionInputs.length; i++) {
    if (petitionInputs[i].value.length < 2) {
      containsErrors = true;
      petitionInputs[i].classList.add('error');
    } else {
      petitionInputs[i].classList.remove('error');
    }
  }

  if (!person.email.includes('.com')) {
    petitionInputs[2].classList.add('error');
    containsErrors = true;
  } else {
    petitionInputs[2].classList.remove('error');
  }

  if (!containsErrors) {
    addSignature(person);
    for (let i = 0; i < petitionInputs.length; i++) {
      petitionInputs[i].value = "";
    }
    toggleModal(person);
  } else {
    // Clear modal content and hide the modal if there are errors
    let modal = document.getElementById("thanks-modal");
    let modalContent = document.getElementById("thanks-modal-content");
    modalContent.textContent = ""; // Clear modal content
    modal.style.display = "none"; // Hide modal
  }
};



