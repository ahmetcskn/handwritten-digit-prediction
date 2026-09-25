const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const result = document.getElementById('result');
const predictButton = document.getElementById('predictButton');
let drawing = false;
let canvasVersion = 0;

ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = 'white';

function point(event) {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * canvas.width / rect.width,
            (event.clientY - rect.top) * canvas.height / rect.height];
}
canvas.addEventListener('pointerdown', event => {
    if (!event.isPrimary || event.button !== 0) return;
    canvasVersion += 1;
    drawing = true;
    canvas.setPointerCapture(event.pointerId);
    ctx.beginPath();
    ctx.moveTo(...point(event));
});
canvas.addEventListener('pointermove', event => {
    if (!drawing) return;
    ctx.lineTo(...point(event));
    ctx.stroke();
});
['pointerup', 'pointercancel', 'lostpointercapture'].forEach(name => {
    canvas.addEventListener(name, () => { drawing = false; });
});

function resetPrediction() {
    if (!predictButton) return;
    document.getElementById('predictedDigit').textContent = '—';
    document.getElementById('confidenceLabel').textContent = 'Waiting for your drawing';
    document.getElementById('confidenceBar').style.width = '0%';
}
document.getElementById('clearButton').addEventListener('click', () => {
    canvasVersion += 1;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetPrediction();
    result.textContent = predictButton ? 'Draw a digit and select Predict digit to get started.' : 'Draw the requested digit to continue.';
});

predictButton?.addEventListener('click', async () => {
    const version = canvasVersion;
    predictButton.disabled = true;
    resetPrediction();
    result.textContent = 'Reading your drawing…';
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: new URLSearchParams({ image: canvas.toDataURL('image/png') }),
        });
        const data = await response.json();
        if (version !== canvasVersion) return;
        if (data.message) {
            result.textContent = data.message;
            return;
        }
        if (!response.ok) throw new Error('Prediction request failed');
        const confidence = (data.confidence * 100).toFixed(1);
        document.getElementById('predictedDigit').textContent = data.digit;
        document.getElementById('confidenceLabel').textContent = `${confidence}% confidence`;
        document.getElementById('confidenceBar').style.width = `${confidence}%`;
        result.textContent = `Prediction: ${data.digit}. Confidence reflects the model’s estimate, not a guarantee.`;
    } catch {
        if (version === canvasVersion) result.textContent = 'Could not read the drawing. Please try again.';
    } finally {
        predictButton.disabled = false;
    }
});

document.getElementById('trainForm')?.addEventListener('submit', () => {
    document.getElementById('imageData').value = canvas.toDataURL('image/png');
    document.getElementById('doneButton').disabled = true;
    result.textContent = 'Saving your drawing. Final training may take a moment…';
});
