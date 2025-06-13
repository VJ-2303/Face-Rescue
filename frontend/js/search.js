// Search functionality
class SearchManager {
    constructor() {
        this.currentPhoto = null;
        this.initializeElements();
    }
    
    initializeElements() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('photo-upload');
        const searchBtn = document.getElementById('search-face');
        const clearBtn = document.getElementById('clear-preview');
        
        // File upload handlers
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // Drag and drop
        setupDragAndDrop(uploadArea, (files) => this.handleFilesDrop(files));
        
        // Search and clear buttons
        searchBtn.addEventListener('click', () => this.performSearch());
        clearBtn.addEventListener('click', () => this.clearPreview());
    }
    
    handleFileSelect(event) {
        const files = Array.from(event.target.files);
        this.handleFiles(files);
    }
    
    handleFilesDrop(files) {
        this.handleFiles(files);
    }
    
    handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        if (!isValidImageFile(file)) {
            showToast('Please select a valid image file (JPEG, PNG, WebP)', 'error');
            return;
        }
        
        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showToast('File size too large. Please select an image under 10MB', 'error');
            return;
        }
        
        this.currentPhoto = file;
        this.showPreview(file);
    }
    
    async showPreview(file) {
        const previewSection = document.getElementById('preview-section');
        const previewImage = document.getElementById('preview-image');
        
        try {
            // Resize image if necessary
            const resizedFile = await resizeImage(file, 800, 600, 0.8);
            this.currentPhoto = resizedFile;
            
            const imageUrl = URL.createObjectURL(resizedFile);
            previewImage.src = imageUrl;
            previewSection.style.display = 'block';
            
            // Scroll to preview
            previewSection.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error processing image:', error);
            showToast('Error processing image', 'error');
        }
    }
    
    clearPreview() {
        const previewSection = document.getElementById('preview-section');
        const previewImage = document.getElementById('preview-image');
        const resultsSection = document.getElementById('search-results');
        const fileInput = document.getElementById('photo-upload');
        
        previewSection.style.display = 'none';
        resultsSection.style.display = 'none';
        previewImage.src = '';
        fileInput.value = '';
        this.currentPhoto = null;
        
        showToast('Preview cleared', 'success');
    }
    
    async performSearch() {
        if (!this.currentPhoto) {
            showToast('Please select or capture a photo first', 'error');
            return;
        }
        
        try {
            showLoading('Searching for matching student...');
            
            const result = await api.searchFace(this.currentPhoto);
            
            hideLoading();
            this.displaySearchResults(result);
            
        } catch (error) {
            hideLoading();
            handleError(error, 'Search failed. Please try again.');
        }
    }
    
    displaySearchResults(result) {
        const resultsSection = document.getElementById('search-results');
        const resultTitle = document.getElementById('result-title');
        const resultConfidence = document.getElementById('result-confidence');
        const resultContent = document.getElementById('result-content');
        
        resultsSection.style.display = 'block';
        
        if (result.match_found) {
            const student = result.student;
            const confidence = result.confidence;
            
            resultTitle.innerHTML = `<i class=\"fas fa-check-circle\" style=\"color: #4CAF50;\"></i> Student Found!`;
            resultConfidence.innerHTML = `
                <span class=\"confidence-badge ${getConfidenceClass(confidence)}\">
                    ${formatConfidence(confidence)} - ${getMatchQuality(confidence)} Match
                </span>
            `;
            
            resultContent.innerHTML = generateStudentCard(student);
            
            showToast(`Student identified: ${student.name}`, 'success');
        } else {
            resultTitle.innerHTML = `<i class=\"fas fa-times-circle\" style=\"color: #F44336;\"></i> No Match Found`;
            resultConfidence.innerHTML = '';
            
            resultContent.innerHTML = `
                <div class=\"no-match-message\">
                    <div style=\"text-align: center; padding: 2rem;\">
                        <i class=\"fas fa-search\" style=\"font-size: 3rem; color: #666; margin-bottom: 1rem;\"></i>
                        <h3 style=\"margin-bottom: 1rem; color: #333;\">No Matching Student Found</h3>
                        <p style=\"color: #666; margin-bottom: 2rem;\">${result.message}</p>
                        ${result.suggestion ? `<p style=\"color: #888; font-style: italic;\">${result.suggestion}</p>` : ''}
                        <div style=\"margin-top: 2rem;\">
                            <button class=\"btn btn-primary\" onclick=\"searchManager.showBatchResults()\">
                                <i class=\"fas fa-list\"></i>
                                Show Similar Students
                            </button>
                        </div>
                    </div>
                </div>
            `;
            
            showToast(result.message, 'warning');
        }
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    async showBatchResults() {
        if (!this.currentPhoto) {
            showToast('Please select or capture a photo first', 'error');
            return;
        }
        
        try {
            showLoading('Finding similar students...');
            
            const result = await api.batchSearchFace(this.currentPhoto, 5);
            
            hideLoading();
            this.displayBatchResults(result);
            
        } catch (error) {
            hideLoading();
            handleError(error, 'Failed to find similar students.');
        }
    }
    
    displayBatchResults(result) {
        const resultContent = document.getElementById('result-content');
        
        if (result.matches && result.matches.length > 0) {
            let html = `
                <div class=\"batch-results\">
                    <h3 style=\"margin-bottom: 1.5rem; color: #333;\">
                        <i class=\"fas fa-users\"></i>
                        Similar Students (${result.total_matches} found)
                    </h3>
                    <div class=\"similar-students-grid\">
            `;
            
            result.matches.forEach(match => {
                const initials = getInitials(match.name);
                html += `
                    <div class=\"similar-student-card\" onclick=\"studentsManager.showStudentDetails('${match.id}')\">
                        <div class=\"similar-student-header\">
                            <div class=\"student-card-avatar\">${initials}</div>
                            <div>
                                <h4>${match.name}</h4>
                                <p>${match.school || 'School not specified'}</p>
                            </div>
                        </div>
                        <div class=\"similarity-info\">
                            <span class=\"confidence-badge ${getConfidenceClass(match.confidence)}\">
                                ${formatConfidence(match.confidence)}
                            </span>
                            <p>${match.match_quality} Match</p>
                        </div>
                        <div class=\"contact-info\">
                            <p><i class=\"fas fa-user\"></i> ${match.emergency_contact.guardian_name}</p>
                            <p><i class=\"fas fa-phone\"></i> ${formatPhoneNumber(match.emergency_contact.phone)}</p>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                    <div style=\"text-align: center; margin-top: 2rem;\">
                        <p style=\"color: #666; font-style: italic;\">
                            These students have some facial similarity but didn't meet the confidence threshold for automatic identification.
                        </p>
                    </div>
                </div>
            `;
            
            resultContent.innerHTML = html;
        } else {
            resultContent.innerHTML = `
                <div style=\"text-align: center; padding: 2rem;\">
                    <i class=\"fas fa-users-slash\" style=\"font-size: 3rem; color: #666; margin-bottom: 1rem;\"></i>
                    <h3 style=\"margin-bottom: 1rem; color: #333;\">No Similar Students Found</h3>
                    <p style=\"color: #666;\">No students in the database have sufficient facial similarity to the uploaded photo.</p>
                </div>
            `;
        }
    }
    
    // Handle camera capture integration
    async handleCameraCapture() {
        if (cameraManager.isRunning()) {
            const capturedFile = await cameraManager.capturePhoto();
            if (capturedFile) {
                this.currentPhoto = capturedFile;
            }
        }
    }
}

// Global search manager
const searchManager = new SearchManager();

// Add CSS for batch results
const style = document.createElement('style');
style.textContent = `
    .similar-students-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
    }
    
    .similar-student-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .similar-student-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    .similar-student-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .similarity-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: white;
        border-radius: 4px;
    }
    
    .contact-info p {
        margin: 0.25rem 0;
        font-size: 0.9rem;
        color: #666;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .contact-info i {
        width: 16px;
        color: #667eea;
    }
`;
document.head.appendChild(style);
