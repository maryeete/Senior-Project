window.onload = function () {
    // Retrieve the data passed from Flask (stored in hidden divs)
    const sentimentData = JSON.parse(document.getElementById('sentimentData').innerText);
    const emotionData = JSON.parse(document.getElementById('emotionData').innerText);

    // Prepare sentiment chart data
    const sentimentLabels = Object.keys(sentimentData);
    const sentimentValues = Object.values(sentimentData);

    // Prepare emotion chart data
    const dominantEmotionLabels = Object.keys(emotionData.dominant_emotion);
    const dominantEmotionValues = Object.values(emotionData.dominant_emotion);
    const otherEmotionLabels = Object.keys(emotionData.other_emotions);
    const otherEmotionValues = Object.values(emotionData.other_emotions);

    // Create sentiment pie chart
    const sentimentPieChart = new Chart(document.getElementById('sentimentPieChart').getContext('2d'), {
        type: 'pie',
        data: {
            labels: sentimentLabels,
            datasets: [{
                data: sentimentValues,
                backgroundColor: ['#FF5733', '#33FF57', '#3357FF'], // Example colors
            }]
        }
    });

    // Create emotion pie chart for dominant emotions
    const emotionPieChart = new Chart(document.getElementById('emotionPieChart').getContext('2d'), {
        type: 'pie',
        data: {
            labels: dominantEmotionLabels,
            datasets: [{
                data: dominantEmotionValues,
                backgroundColor: ['#FF5733', '#33FF57', '#3357FF', '#FFD700'], // Example colors
            }]
        }
    });

    // Create emotion bar chart for other emotions
    const emotionBarChart = new Chart(document.getElementById('emotionBarChart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: otherEmotionLabels,
            datasets: [{
                label: 'Other Emotions',
                data: otherEmotionValues,
                backgroundColor: '#FF5733',
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
