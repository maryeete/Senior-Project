// Auto-hide flash messages after 5 seconds
document.addEventListener("DOMContentLoaded", () => {
    const alerts = document.querySelectorAll(".alert");
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = "0";
            setTimeout(() => alert.remove(), 500); // Removes the alert after fading out
        }, 5000);
    });
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
