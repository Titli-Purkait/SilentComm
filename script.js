const video = document.getElementById('video');
const captureBtn = document.getElementById('captureBtn');
const emotionDisplay = document.getElementById('emotion');

// start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => {
    console.error('Webcam error:', err);
    emotionDisplay.textContent = 'Cannot access camera.';
  });

// grab frame as JPEG dataURL
function getFrameDataURL() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg');
}

// send to Flask for prediction
captureBtn.addEventListener('click', () => {
  emotionDisplay.textContent = 'Predicting…';
  const dataURL = getFrameDataURL();

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataURL })
  })
  .then(res => res.json())
  .then(json => {
    emotionDisplay.textContent = json.error
      ? 'Error: ' + json.error
      : 'Emotion: ' + json.emotion;
  })
  .catch(err => {
    console.error('Fetch error:', err);
    emotionDisplay.textContent = 'Prediction failed.';
  });
});
