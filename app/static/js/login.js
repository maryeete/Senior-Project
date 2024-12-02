document.addEventListener("DOMContentLoaded", () => {
    const alerts = document.querySelectorAll(".alert");
    alerts.forEach(alert => {
        // Auto-hide flash messages after 5 seconds
        setTimeout(() => {
            alert.style.opacity = "0";
            setTimeout(() => alert.remove(), 500); // Removes the alert after fading out
        }, 5000);

        // Add functionality for close button
        const closeButton = alert.querySelector(".close");
        if (closeButton) {
            closeButton.addEventListener("click", () => {
                alert.style.opacity = "0";
                setTimeout(() => alert.remove(), 500); // Removes the alert after fade-out
            });
        }
    });

    // Password visibility toggle
    const passwordInput = document.getElementById("password");
    const togglePassword = document.getElementById("toggle-password");
    togglePassword.addEventListener("click", () => {
        const type = passwordInput.getAttribute("type") === "password" ? "text" : "password";
        passwordInput.setAttribute("type", type);
        togglePassword.textContent = type === "password" ? "🙈" : "🙈👁️";
    });

    // Ripple effect for the submit button
    const button = document.querySelector(".btn");
    button.addEventListener("click", (e) => {
        const ripple = document.createElement("span");
        const rect = button.getBoundingClientRect();

        // Set ripple size and position
        ripple.style.width = ripple.style.height = Math.max(rect.width, rect.height) + "px";
        ripple.style.left = e.clientX - rect.left - ripple.offsetWidth / 2 + "px";
        ripple.style.top = e.clientY - rect.top - ripple.offsetHeight / 2 + "px";

        ripple.classList.add("ripple");
        button.appendChild(ripple);

        // Remove ripple after animation
        setTimeout(() => {
            ripple.remove();
        }, 600);
    });
});
