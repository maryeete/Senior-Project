// Function to show a specific section
function showSection(sectionId) {
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => section.classList.remove('active'));

    const selectedSection = document.getElementById(sectionId);
    selectedSection.classList.add('active');
}

// Initialize the page with Video Analysis section active
document.addEventListener('DOMContentLoaded', () => {
    showSection('video-section');
});
