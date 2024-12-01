// analyze_video.js

// Simulate real-time emotion detection updates
document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('emotionVideo');

    // Placeholder for updates (simulate detecting emotions)
    setInterval(() => {
        const emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral'];
        const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];

        // Dynamically update the alt text for the video
        videoElement.alt = `Detected Emotion: ${randomEmotion}`;
        console.log(`Current Emotion: ${randomEmotion}`); // Log for debugging
    }, 3000); // Update every 3 seconds
});
