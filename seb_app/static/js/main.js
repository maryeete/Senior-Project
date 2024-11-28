document.addEventListener('DOMContentLoaded', function() {
    loadDevices();
    hideAllSections();
});

async function loadDevices() {
    try {
        const response = await fetch('/get_devices');
        const devices = await response.json();
        
        const cameraSelect = document.getElementById('cameraSelect');
        const microphoneSelect = document.getElementById('microphoneSelect');
        
        // Populate camera options
        devices.cameras.forEach(camera => {
            const option = document.createElement('option');
            option.value = camera.id;
            option.textContent = camera.name;
            cameraSelect.appendChild(option);
        });
        
        // Populate microphone options
        devices.microphones.forEach(mic => {
            const option = document.createElement('option');
            option.value = mic.id;
            option.textContent = mic.name;
            microphoneSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading devices:', error);
    }
}

function hideAllSections() {
    const sections = document.getElementsByClassName('analysis-section');
    for (let section of sections) {
        section.style.display = 'none';
    }
}

function showSection(sectionId) {
    hideAllSections();
    document.getElementById(sectionId).style.display = 'block';
}

// Image Analysis
async function analyzeImage() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/analyze_image', {
            method: 'POST',
            body: formData
        });
        
        const results = await response.json();
        displayImageResults(results);
    } catch (error) {
        console.error('Error analyzing image:', error);
    }
}

let videoStream = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// Video Analysis
function startVideo() {
    const cameraSelect = document.getElementById('cameraSelect');
    const videoFeed = document.getElementById('videoFeed');
    const cameraId = cameraSelect.value || 0;
    
    fetch('/start_video', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ camera_id: parseInt(cameraId) })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            alert('Error starting video: ' + data.error);
            return;
        }
        videoFeed.style.display = 'block';
        videoFeed.src = data.stream_url;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error starting video');
    });
}

function stopVideo() {
    const videoFeed = document.getElementById('videoFeed');
    videoFeed.src = '';
    videoFeed.style.display = 'none';
}

// Audio Analysis
async function startRecording() {
    const micSelect = document.getElementById('microphoneSelect');
    const micId = micSelect.value || 0;
    
    try {
        // Start server-side recording
        await fetch('/start_audio_recording', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ microphone_id: micId })
        });
        
        // Start client-side recording for file upload
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        document.getElementById('startRecording').disabled = true;
        document.getElementById('stopRecording').disabled = false;
        
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Error starting recording. Please check your microphone permissions.');
    }
}

async function stopRecording() {
    if (!isRecording) return;
    
    try {
        // Stop client-side recording
        mediaRecorder.stop();
        isRecording = false;
        
        // Stop server-side recording and get results
        const response = await fetch('/stop_audio_recording', {
            method: 'POST'
        });
        
        const data = await response.json();
        if (data.results) {
            displayAudioResults(data.results);
        }
        
        // Update UI
        document.getElementById('startRecording').disabled = false;
        document.getElementById('stopRecording').disabled = true;
        
    } catch (error) {
        console.error('Error stopping recording:', error);
        alert('Error stopping recording.');
    }
}

async function analyzeAudioFile() {
    const input = document.getElementById('audioInput');
    const file = input.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('audio', file);
    
    try {
        const response = await fetch('/analyze_audio', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.results) {
            displayAudioResults(data.results);
        }
    } catch (error) {
        console.error('Error analyzing audio:', error);
        alert('Error analyzing audio file.');
    }
}

function displayImageResults(results) {
    const resultsDiv = document.getElementById('imageResults');
    resultsDiv.innerHTML = '';
    
    results.forEach(face => {
        const faceDiv = document.createElement('div');
        faceDiv.className = 'face-result';
        
        let emotionsHtml = '<h3>Detected Emotions:</h3><ul>';
        for (const [emotion, score] of Object.entries(face.emotions)) {
            emotionsHtml += `<li>${emotion}: ${(score * 100).toFixed(2)}%</li>`;
        }
        emotionsHtml += '</ul>';
        
        faceDiv.innerHTML = emotionsHtml;
        resultsDiv.appendChild(faceDiv);
    });
}

function displayAudioResults(results) {
    const resultsDiv = document.getElementById('audioResults');
    resultsDiv.innerHTML = '<h3>Audio Analysis Results:</h3><ul>';
    
    for (const [emotion, probability] of Object.entries(results)) {
        const percentage = (probability * 100).toFixed(2);
        resultsDiv.innerHTML += `<li>${emotion}: ${percentage}%</li>`;
    }
    
    resultsDiv.innerHTML += '</ul>';
}

// Add these functions for combined analysis
let combinedStream = null;

async function startCombined() {
    const cameraSelect = document.getElementById('cameraSelect');
    const micSelect = document.getElementById('microphoneSelect');
    const cameraId = cameraSelect.value || 0;
    const micId = micSelect.value || 0;
    
    try {
        // Request microphone permission
        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Start server-side combined analysis
        const response = await fetch('/start_combined', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                camera_id: cameraId,
                microphone_id: micId
            })
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            const combinedVideo = document.getElementById('combinedVideo');
            combinedVideo.src = data.stream_url;
            combinedStream = audioStream;
            
            // Update UI
            document.getElementById('startCombined').disabled = true;
            document.getElementById('stopCombined').disabled = false;
        }
    } catch (error) {
        console.error('Error starting combined analysis:', error);
        alert('Error starting combined analysis. Please check your device permissions.');
    }
}

async function stopCombined() {
    try {
        // Stop client-side streams
        if (combinedStream) {
            combinedStream.getTracks().forEach(track => track.stop());
            combinedStream = null;
        }
        
        // Stop server-side analysis
        await fetch('/stop_combined', {
            method: 'POST'
        });
        
        // Clear video
        const combinedVideo = document.getElementById('combinedVideo');
        combinedVideo.src = '';
        
        // Update UI
        document.getElementById('startCombined').disabled = false;
        document.getElementById('stopCombined').disabled = true;
        
    } catch (error) {
        console.error('Error stopping combined analysis:', error);
        alert('Error stopping combined analysis.');
    }
}
