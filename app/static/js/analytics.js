// Retrieve chart data from the HTML element
const chartDataElement = document.getElementById('chart-data');
const labels = JSON.parse(chartDataElement.getAttribute('data-labels'));
const data = JSON.parse(chartDataElement.getAttribute('data-data'));

// Pie Chart
const pieData = {
    labels: labels,
    datasets: [{
        data: data,
        backgroundColor: ['#ff6384', '#36a2eb', '#cc65fe', '#ffce56', '#ff5733'],
        hoverBackgroundColor: ['#ff4c60', '#1f76c3', '#9f4ce0', '#ffac28', '#ff5733'],
    }]
};

new Chart(document.getElementById('pieChart'), {
    type: 'pie',
    data: pieData
});

// Line Chart
const lineData = {
    labels: labels,
    datasets: [{
        label: 'Emotion Count Trend',
        data: data,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
    }]
};

new Chart(document.getElementById('lineChart'), {
    type: 'line',
    data: lineData
});

// Bar Chart
const barData = {
    labels: labels,
    datasets: [{
        label: 'Emotion Count',
        data: data,
        backgroundColor: '#42A5F5',
        borderColor: '#1E88E5',
        borderWidth: 1
    }]
};

new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: barData,
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});
