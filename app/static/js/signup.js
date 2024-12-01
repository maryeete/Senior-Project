// Background Animation Script
document.addEventListener("DOMContentLoaded", () => {
    const overlay = document.querySelector(".overlay");
    createStars(overlay, 100);

    function createStars(container, count) {
        for (let i = 0; i < count; i++) {
            const star = document.createElement("div");
            star.className = "star";
            star.style.top = `${Math.random() * 100}%`;
            star.style.left = `${Math.random() * 100}%`;
            star.style.animationDuration = `${Math.random() * 5 + 3}s`;
            container.appendChild(star);
        }
    }

    // Form Validation and Interactivity
    const form = document.querySelector("form");
    const inputs = document.querySelectorAll(".form-group input");
    const passwordField = document.querySelector("input[type='password']");
    
    // Add password visibility toggle
    const togglePasswordBtn = document.createElement("span");
    togglePasswordBtn.textContent = "🙈";
    togglePasswordBtn.style.cursor = "pointer";
    togglePasswordBtn.style.marginLeft = "-30px";
    togglePasswordBtn.style.position = "relative";
    passwordField.parentNode.style.position = "relative";
    passwordField.parentNode.appendChild(togglePasswordBtn);

    togglePasswordBtn.addEventListener("click", () => {
        if (passwordField.type === "password") {
            passwordField.type = "text";
            togglePasswordBtn.textContent = "👁️";
        } else {
            passwordField.type = "password";
            togglePasswordBtn.textContent = "🙈";
        }
    });

    // Input validation
    form.addEventListener("submit", (e) => {
        let valid = true;
        inputs.forEach(input => {
            if (!input.value.trim()) {
                valid = false;
                input.style.borderColor = "red";
            } else {
                input.style.borderColor = "rgba(255, 255, 255, 0.3)";
            }
        });

        // Check if all fields are filled correctly
        if (!valid) {
            e.preventDefault();
            alert("Please fill in all fields!");
        }
    });
});
