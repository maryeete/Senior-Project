function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.analysis-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById(sectionId).classList.add('active');
    
    // Add active class to clicked button
    const button = Array.from(document.querySelectorAll('.analysis-button'))
        .find(btn => btn.textContent.toLowerCase().includes(sectionId.split('-')[0]));
    if (button) button.classList.add('active');
}

// Show video section by default
document.addEventListener('DOMContentLoaded', () => {
    showSection('video-section');
});


// Device loading functions
async function loadDevices() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        
        // Load cameras
        const cameras = devices.filter(device => device.kind === 'videoinput');
        const cameraSelect = document.getElementById('cameraSelect');
        cameraSelect.innerHTML = cameras.length 
            ? cameras.map((camera, idx) => 
                `<option value="${idx}">${camera.label || `Camera ${idx + 1}`}</option>`).join('')
            : '<option value="">No cameras found</option>';

        // Load microphones
        const microphones = devices.filter(device => device.kind === 'audioinput');
        const micSelect = document.getElementById('microphoneSelect');
        micSelect.innerHTML = microphones.length
            ? microphones.map((mic, idx) => 
                `<option value="${idx}">${mic.label || `Microphone ${idx + 1}`}</option>`).join('')
            : '<option value="">No microphones found</option>';
    } catch (error) {
        console.error('Error loading devices:', error);
    }
}

// Request permissions and load devices
async function initialize() {
    try {
        // Request camera and microphone permissions
        await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        await loadDevices();
    } catch (error) {
        console.error('Error initializing:', error);
    }
}

// Video functions
async function startVideo() {
    const cameraId = document.getElementById('cameraSelect').value;
    const videoFeed = document.getElementById('videoFeed');
    
    try {
        const response = await fetch('/start_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ camera_id: cameraId })
        });
        
        const data = await response.json();
        if (data.stream_url) {
            videoFeed.src = data.stream_url;
            videoFeed.style.display = 'block';
        }
    } catch (error) {
        console.error('Error starting video:', error);
    }
}

function stopVideo() {
    const videoFeed = document.getElementById('videoFeed');
    videoFeed.src = '';
    videoFeed.style.display = 'none';
    
    fetch('/stop_combined', { method: 'POST' })
        .catch(error => console.error('Error stopping video:', error));
}

// Image analysis functions
function analyzeImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select an image file');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    fetch('/analyze_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const emotionCards = document.getElementById('emotionCards');
        if (data.results && data.results.length > 0) {
            emotionCards.innerHTML = data.results.map(result => `
                <div class="emotion-card">
                    <div class="dominant-emotion">
                        <h4>Primary Emotion</h4>
                        <div class="emotion-bar-container">
                            <div class="emotion-label">
                                <span>${result.dominant_emotion.name}</span>
                                <span>${result.dominant_emotion.confidence}%</span>
                            </div>
                            <div class="emotion-bar">
                                <div class="emotion-bar-fill" style="width: ${result.dominant_emotion.confidence}%"></div>
                            </div>
                        </div>
                    </div>
                    ${result.other_emotions.length ? `
                        <div class="other-emotions">
                            <h4>Other Emotions</h4>
                            ${result.other_emotions.map(emotion => `
                                <div class="emotion-bar-container">
                                    <div class="emotion-label">
                                        <span>${emotion.name}</span>
                                        <span>${emotion.confidence}%</span>
                                    </div>
                                    <div class="emotion-bar">
                                        <div class="emotion-bar-fill" style="width: ${emotion.confidence}%"></div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `).join('');
        } else {
            emotionCards.innerHTML = `
                <div class="error-message">
                    <p>${data.message || 'No emotions detected'}</p>
                    ${data.suggestions ? `
                        <ul>
                            ${data.suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
                        </ul>
                    ` : ''}
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('emotionCards').innerHTML = 
            '<div class="error-message">Error analyzing image</div>';
    });
}

// Add these variables at the top of your script section
let mediaRecorder = null;
let videoChunks = [];
let recordingTimer = null;
let recordingStartTime = null;
let isRecording = false;

async function toggleRecording() {
    const recordButton = document.getElementById('recordingButton');
    
    if (!isRecording) {
        // Start Recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true,
            });
            console.log("a"); 
            mediaRecorder = new MediaRecorder(stream);
            videoChunks = [];
            console.log("b"); 
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    videoChunks.push(event.data);
                }
            };
            console.log("c"); 
            mediaRecorder.start(1000);

            // Start the timer
            recordingStartTime = Date.now();
            const timerDisplay = document.getElementById('recordingTimer');
            const recordingStatus = document.getElementById('recordingStatus');
            
            recordingStatus.style.display = 'block';
            recordingTimer = setInterval(() => {
                const elapsed = Date.now() - recordingStartTime;
                const seconds = Math.floor((elapsed / 1000) % 60).toString().padStart(2, '0');
                const minutes = Math.floor((elapsed / 1000 / 60)).toString().padStart(2, '0');
                timerDisplay.textContent = `${minutes}:${seconds}`;
            }, 1000);

            // Update UI
            recordButton.textContent = 'Stop Recording';
            recordButton.classList.add('recording');
            isRecording = true;

        } catch (error) {
            console.error('Detailed recording error:', error);
            alert('Error starting recording: ' + error.message);
        }
    } else {
        // Stop Recording
        try {
            mediaRecorder.stop();
            
            // Stop the timer
            if (recordingTimer) {
                clearInterval(recordingTimer);
                recordingTimer = null;
            }
            
            // Hide recording status
            document.getElementById('recordingStatus').style.display = 'none';

            // Process the recording
            await new Promise(resolve => {
                mediaRecorder.onstop = async () => {
                    const videoBlob = new Blob(videoChunks, { type: 'video/mp4' });
                    const formData = new FormData();
                    formData.append('video', videoBlob, 'recorded_video.mp4');
                    
                    document.getElementById('uploadingMessage').style.display = 'block';

                    try {
                        const response = await fetch('/upload_video', {
                            method: 'POST',
                            body: formData
                        });

                        const data = await response.json();
                        if (data.status === 'success') {
                            alert('Video uploaded successfully!');
                        } else {
                            alert('Error uploading video: ' + data.error);
                        }
                    } catch (error) {
                        console.error('Error uploading video:', error);
                        alert('Error uploading video');
                    } finally {
                        // Hide uploading message after the upload process finishes
                        document.getElementById('uploadingMessage').style.display = 'none';
                    }
                    
                    resolve();
                };
            });

            // Clean up
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            mediaRecorder = null;
            videoChunks = [];

            // Update UI
            recordButton.textContent = 'Start Recording';
            recordButton.classList.remove('recording');
            isRecording = false;

        } catch (error) {
            console.error('Error stopping recording:', error);
            alert('Error stopping recording.');
        }
    }
}


// Add event listener to the button
document.getElementById('recordingButton').addEventListener('click', toggleRecording);

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initialize);

// Add this after your existing toggleRecording function
document.getElementById('audioFileInput').addEventListener('change', async function(event) {
    const file = event.target.files[0];
    const fileNameDisplay = document.getElementById('fileName');
    const audioResults = document.getElementById('audioResults');
    
    if (file) {
        try {
            // Clear previous results
            while (audioResults.firstChild) {
                audioResults.removeChild(audioResults.firstChild);
            }
            
            fileNameDisplay.textContent = file.name;
            
            const formData = new FormData();
            formData.append('audio', file);
            
            // Show loading state
            audioResults.innerHTML = '<div class="loading">Analyzing audio...</div>';
            
            const response = await fetch('/analyze_audio_file', {
                method: 'POST',
                body: formData
            });
            
            // Clear loading state
            audioResults.innerHTML = '';
            
            const data = await response.json();
            
            if (data.error) {
                audioResults.innerHTML = `
                    <div class="error-message">
                        ${data.error}
                    </div>
                `;
                return;
            }
            
            if (data.dominant_emotion && data.all_emotions) {
                audioResults.innerHTML = `
                    <div class="emotion-card">
                        <div class="analysis-info">
                            <span class="file-badge">Uploaded File: ${file.name}</span>
                            <span class="timestamp">Analyzed at: ${new Date().toLocaleTimeString()}</span>
                        </div>
                        <div class="dominant-emotion">
                            <h4>Primary Emotion</h4>
                            <div class="emotion-bar-container">
                                <div class="emotion-label">
                                    <span>${data.dominant_emotion.emotion}</span>
                                    <span>${data.dominant_emotion.confidence}%</span>
                                </div>
                                <div class="emotion-bar">
                                    <div class="emotion-bar-fill" style="width: ${data.dominant_emotion.confidence}%"></div>
                                </div>
                            </div>
                        </div>
                        <div class="other-emotions">
                            <h4>Other Emotions</h4>
                            ${Object.entries(data.all_emotions)
                                .filter(([emotion, confidence]) => 
                                    emotion !== data.dominant_emotion.emotion && confidence > 10)
                                .sort((a, b) => b[1] - a[1])
                                .map(([emotion, confidence]) => `
                                    <div class="emotion-bar-container">
                                        <div class="emotion-label">
                                            <span>${emotion}</span>
                                            <span>${confidence.toFixed(1)}%</span>
                                        </div>
                                        <div class="emotion-bar">
                                            <div class="emotion-bar-fill" style="width: ${confidence}%"></div>
                                        </div>
                                    </div>
                                `).join('')}
                        </div>
                    </div>
                `;
            } else {
                audioResults.innerHTML = `
                    <div class="error-message">
                        No emotion detected in the uploaded file
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error analyzing audio file:', error);
            audioResults.innerHTML = `
                <div class="error-message">
                    Error analyzing audio file: ${error.message}
                </div>
            `;
        } finally {
            // Reset file input and filename display
            event.target.value = '';
            fileNameDisplay.textContent = '';
        }
    }
});

function analyzeText() {
    const textInput = document.getElementById('textInput').value.trim();
    const textResults = document.getElementById('textResults');
    
    // Clear previous results
    textResults.innerHTML = '';

    if (!textInput) {
        alert('Please enter some text.');
        return;
    }

    fetch('/analyze_text', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
            'textInput': textInput
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            const sentiment = data.sentiment;
            textResults.innerHTML = `<div><strong>Sentiment:</strong> ${sentiment}</div>`;
        } else {
            textResults.innerHTML = `<div><strong>Error:</strong> ${data.error}</div>`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        textResults.innerHTML = `<div><strong>Failed to analyze text</strong></div>`;
    });
}

