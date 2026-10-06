const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('imageInput');
const chooseImageButton = document.getElementById('chooseImageButton');
const detectButton = document.getElementById('detectButton');
const resetButton = document.getElementById('resetButton');
const uploadForm = document.getElementById('uploadForm');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');
const originalImage = document.getElementById('originalImage');
const resultImage = document.getElementById('resultImage');
const emptyOriginal = document.getElementById('emptyOriginal');
const emptyResult = document.getElementById('emptyResult');
const statusMessage = document.getElementById('statusMessage');
const downloadLink = document.getElementById('downloadLink');
const resultsTableBody = document.getElementById('resultsTableBody');

const totalObjects = document.getElementById('totalObjects');
const helmetCount = document.getElementById('helmetCount');
const reflectiveJacketCount = document.getElementById('reflectiveJacketCount');
const averageConfidence = document.getElementById('averageConfidence');
const highestConfidence = document.getElementById('highestConfidence');

const MAX_FILE_SIZE = 16 * 1024 * 1024;

const formatConfidence = (value) => Number(value).toFixed(2);

function updateThresholdText() {
  confidenceValue.textContent = Number(confidenceSlider.value).toFixed(2);
}

function showStatus(message, type = 'error') {
  statusMessage.textContent = message;
  statusMessage.className = `status-message ${type}`;
  statusMessage.classList.remove('hidden');
}

function hideStatus() {
  statusMessage.classList.add('hidden');
}

function setOriginalPreview(src) {
  originalImage.src = src;
  originalImage.classList.remove('hidden');
  emptyOriginal.classList.add('hidden');
}

function setResultPreview(src) {
  resultImage.src = src;
  resultImage.classList.remove('hidden');
  emptyResult.classList.add('hidden');
}

function clearOriginalPreview() {
  originalImage.removeAttribute('src');
  originalImage.classList.add('hidden');
  emptyOriginal.classList.remove('hidden');
}

function clearResultPreview() {
  resultImage.removeAttribute('src');
  resultImage.classList.add('hidden');
  emptyResult.classList.remove('hidden');
  downloadLink.classList.add('hidden');
  downloadLink.removeAttribute('href');
}

function resetTable() {
  resultsTableBody.innerHTML = '<tr><td colspan="3" class="empty-table">No detections yet.</td></tr>';
}

function resetStats() {
  totalObjects.textContent = '0';
  helmetCount.textContent = '0';
  reflectiveJacketCount.textContent = '0';
  averageConfidence.textContent = '0.00';
  highestConfidence.textContent = '0.00';
}

function resetUI() {
  fileInput.value = '';
  clearOriginalPreview();
  clearResultPreview();
  resetTable();
  resetStats();
  hideStatus();
}

chooseImageButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (!file) return;
  handleSelectedFile(file);
});

function handleSelectedFile(file) {
  if (!file) {
    showStatus('⚠ Please upload an image.', 'error');
    return;
  }

  const allowed = ['image/jpeg', 'image/png'];
  if (!allowed.includes(file.type)) {
    showStatus('⚠ Please upload a JPG or PNG image.', 'error');
    return;
  }

  if (file.size > MAX_FILE_SIZE) {
    showStatus('⚠ File is too large. Please upload an image under 16 MB.', 'error');
    return;
  }

  const previewUrl = URL.createObjectURL(file);
  setOriginalPreview(previewUrl);
  hideStatus();
}

['dragenter', 'dragover'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    fileInput.files = event.dataTransfer.files;
    handleSelectedFile(file);
  }
});

dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    showStatus('⚠ Please upload an image before detecting.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('image', file);

  const threshold = Number(confidenceSlider.value).toFixed(2);
  detectButton.disabled = true;
  detectButton.textContent = 'Analyzing image...';
  showStatus('Analyzing image...', 'success');

  try {
    const response = await fetch(`/api/detect?conf=${threshold}`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Unable to process the image.');
    }

    if (data.image_url) {
      setResultPreview(data.image_url);
      downloadLink.href = data.image_url;
      downloadLink.classList.remove('hidden');
    }

    updateStats(data.statistics);
    updateDetectionTable(data.detections);
    hideStatus();
    showStatus('Detection complete.', 'success');
  } catch (error) {
    console.error(error);
    showStatus(error.message || '⚠ Unable to process the image.', 'error');
    clearResultPreview();
  } finally {
    detectButton.disabled = false;
    detectButton.textContent = 'Detect';
  }
});

function updateStats(statistics = {}) {
  totalObjects.textContent = statistics.total ?? 0;
  helmetCount.textContent = statistics.helmet ?? 0;
  reflectiveJacketCount.textContent = statistics.reflective_jacket ?? 0;
  averageConfidence.textContent = formatConfidence(statistics.average_confidence ?? 0);
  highestConfidence.textContent = formatConfidence(statistics.highest_confidence ?? 0);
}

function updateDetectionTable(detections = []) {
  if (!detections.length) {
    resetTable();
    return;
  }

  resultsTableBody.innerHTML = detections
    .map((item) => {
      const bbox = item.bbox || {};
      const boxText = `(${Math.round(bbox.x1)},${Math.round(bbox.y1)},${Math.round(bbox.x2)},${Math.round(bbox.y2)})`;
      return `
        <tr>
          <td>${escapeHtml(item.class_name || 'Unknown')}</td>
          <td>${formatConfidence(item.confidence * 100)}%</td>
          <td>${escapeHtml(boxText)}</td>
        </tr>
      `;
    })
    .join('');
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

resetButton.addEventListener('click', () => {
  resetUI();
  fileInput.value = '';
});

confidenceSlider.addEventListener('input', updateThresholdText);
updateThresholdText();
resetUI();
