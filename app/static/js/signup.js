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
});
