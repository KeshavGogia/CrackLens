const uploadForm = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const originalImage = document.getElementById('originalImage');
const maskImage = document.getElementById('maskImage');
const graphImage = document.getElementById('graphImage');
const imagesSection = document.getElementById('imagesSection');
const analysisSection = document.getElementById('analysisSection');
const analysisList = document.getElementById('analysisList');
const errorMsg = document.getElementById('errorMsg');

// Show original image preview
imageInput.addEventListener('change', function() {
  if (this.files && this.files[0]) {
    const reader = new FileReader();
    reader.onload = e => {
      originalImage.src = e.target.result;
      imagesSection.style.display = 'flex';
    };
    reader.readAsDataURL(this.files[0]);
  }
});

// Handle form submit
uploadForm.addEventListener('submit', function(e) {
  e.preventDefault();
  errorMsg.textContent = '';
  analysisSection.style.display = 'none';
  maskImage.src = '';
  graphImage.src = '';

  const file = imageInput.files[0];
  if (!file) {
    errorMsg.textContent = "Please select an image.";
    return;
  }
  // Show loading indicator
  maskImage.alt = 'Loading...';
  graphImage.alt = 'Loading...';

  const formData = new FormData();
  formData.append('image', file);

  fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    body: formData
  })
  .then(async response => {
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Unknown error");
    }
    return response.json();
  })
  .then(data => {
    // Show mask and graph
    maskImage.src = 'data:image/png;base64,' + data.result_image;
    maskImage.alt = 'Predicted Mask';
    graphImage.src = 'data:image/png;base64,' + data.graph_image;
    graphImage.alt = 'Crack Graph';

    imagesSection.style.display = 'flex';

    // Show analysis
    const analysis = data.analysis;
    analysisList.innerHTML = `
      <li><strong>Crack Length:</strong> ${analysis.length_pixels.toFixed(2)} pixels</li>
      <li><strong>Width Mean:</strong> ${analysis.width_stats.mean.toFixed(2)} px</li>
      <li><strong>Width Std Dev:</strong> ${analysis.width_stats.std.toFixed(2)} px</li>
      <li><strong>Width Range (mean ± 2σ):</strong> 
        ${analysis.width_stats.range[0].toFixed(2)} px to ${analysis.width_stats.range[1].toFixed(2)} px
      </li>
      <li><strong>Width Min:</strong> ${analysis.width_stats.min.toFixed(2)} px</li>
      <li><strong>Width Max:</strong> ${analysis.width_stats.max.toFixed(2)} px</li>
    `;
    analysisSection.style.display = 'block';
  })
  .catch(err => {
    errorMsg.textContent = "Error: " + err.message;
    imagesSection.style.display = 'none';
    analysisSection.style.display = 'none';
  });
});
