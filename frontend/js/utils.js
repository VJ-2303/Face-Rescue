// Utility Functions

// Show loading overlay
function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    text.textContent = message;
    overlay.classList.add('active');
}

// Hide loading overlay
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.remove('active');
}

// Show toast notification
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <i class="fas fa-${getToastIcon(type)}"></i>
            <span>${message}</span>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (container.contains(toast)) {
                container.removeChild(toast);
            }
        }, 300);
    }, 5000);
}

function getToastIcon(type) {
    switch (type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        default: return 'info-circle';
    }
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Format phone number for display
function formatPhoneNumber(phone) {
    if (!phone) return '';
    const cleaned = phone.replace(/\D/g, '');
    if (cleaned.length === 10) {
        return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3, 6)}-${cleaned.slice(6)}`;
    }
    return phone;
}

// Get initials from name
function getInitials(name) {
    if (!name) return '?';
    return name.split(' ')
        .map(word => word.charAt(0))
        .join('')
        .toUpperCase()
        .slice(0, 2);
}

// Validate file type
function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    return validTypes.includes(file.type);
}

// Convert file to base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Resize image
function resizeImage(file, maxWidth = 800, maxHeight = 600, quality = 0.8) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = () => {
            // Calculate new dimensions
            let { width, height } = img;
            
            if (width > maxWidth) {
                height = (height * maxWidth) / width;
                width = maxWidth;
            }
            
            if (height > maxHeight) {
                width = (width * maxHeight) / height;
                height = maxHeight;
            }
            
            // Set canvas dimensions
            canvas.width = width;
            canvas.height = height;
            
            // Draw and compress
            ctx.drawImage(img, 0, 0, width, height);
            canvas.toBlob(resolve, 'image/jpeg', quality);
        };
        
        img.src = URL.createObjectURL(file);
    });
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Show modal
function showModal(modalId, content = null) {
    const modal = document.getElementById(modalId);
    if (content) {
        const body = modal.querySelector('.modal-body');
        body.innerHTML = content;
    }
    modal.classList.add('active');
    
    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            hideModal(modalId);
        }
    });
}

// Hide modal
function hideModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('active');
}

// Format confidence score
function formatConfidence(confidence) {
    return `${(confidence * 100).toFixed(1)}%`;
}

// Get confidence class
function getConfidenceClass(confidence) {
    if (confidence >= 0.85) return 'confidence-excellent';
    if (confidence >= 0.75) return 'confidence-very-good';
    if (confidence >= 0.65) return 'confidence-good';
    if (confidence >= 0.55) return 'confidence-fair';
    return 'confidence-poor';
}

// Get match quality text
function getMatchQuality(confidence) {
    if (confidence >= 0.85) return 'Excellent';
    if (confidence >= 0.75) return 'Very Good';
    if (confidence >= 0.65) return 'Good';
    if (confidence >= 0.55) return 'Fair';
    return 'Poor';
}

// Validate form data
function validateForm(formElement) {
    const errors = [];
    const requiredFields = formElement.querySelectorAll('[required]');
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            errors.push(`${field.previousElementSibling.textContent} is required`);
            field.style.borderColor = '#F44336';
        } else {
            field.style.borderColor = '#ddd';
        }
    });
    
    return errors;
}

// Clear form
function clearForm(formElement) {
    const inputs = formElement.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        if (input.type === 'file') {
            input.value = '';
        } else if (input.type === 'checkbox' || input.type === 'radio') {
            input.checked = false;
        } else {
            input.value = '';
        }
        input.style.borderColor = '#ddd';
    });
}

// Generate student card HTML
function generateStudentCard(student) {
    const initials = getInitials(student.name);
    const formattedPhone = formatPhoneNumber(student.emergency_contact.phone);
    const altPhone = student.emergency_contact.alternate_phone ? 
        formatPhoneNumber(student.emergency_contact.alternate_phone) : null;
    
    return `
        <div class="student-card">
            <div class="student-avatar">
                ${initials}
            </div>
            <div class="student-info">
                <h3>${student.name}</h3>
                <div class="info-grid">
                    ${student.age ? `<div class="info-item"><i class="fas fa-birthday-cake"></i><span>Age: ${student.age}</span></div>` : ''}
                    ${student.gender ? `<div class="info-item"><i class="fas fa-venus-mars"></i><span>${student.gender}</span></div>` : ''}
                    ${student.school ? `<div class="info-item"><i class="fas fa-school"></i><span>${student.school}</span></div>` : ''}
                    ${student.grade ? `<div class="info-item"><i class="fas fa-graduation-cap"></i><span>Grade ${student.grade}</span></div>` : ''}
                    <div class="info-item">
                        <i class="fas fa-user"></i>
                        <span>${student.emergency_contact.guardian_name} (${student.emergency_contact.relationship})</span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-phone"></i>
                        <span>${formattedPhone}</span>
                    </div>
                    ${altPhone ? `<div class="info-item"><i class="fas fa-phone-alt"></i><span>${altPhone}</span></div>` : ''}
                    ${student.emergency_contact.address ? `<div class="info-item"><i class="fas fa-map-marker-alt"></i><span>${student.emergency_contact.address}</span></div>` : ''}
                </div>
                ${student.medical_info ? generateMedicalInfo(student.medical_info) : ''}
                ${student.notes ? `<div class="info-item mt-2"><i class="fas fa-sticky-note"></i><span>${student.notes}</span></div>` : ''}
            </div>
            <div class="emergency-actions">
                <a href="tel:${student.emergency_contact.phone}" class="emergency-btn call-btn">
                    <i class="fas fa-phone"></i>
                    Call Guardian
                </a>
                <a href="sms:${student.emergency_contact.phone}" class="emergency-btn sms-btn">
                    <i class="fas fa-sms"></i>
                    Send SMS
                </a>
            </div>
        </div>
    `;
}

// Generate medical info HTML
function generateMedicalInfo(medicalInfo) {
    if (!medicalInfo) return '';
    
    let html = '<div class="medical-info mt-2">';
    
    if (medicalInfo.conditions && medicalInfo.conditions.length > 0) {
        html += `<div class="info-item"><i class="fas fa-heartbeat"></i><span>Conditions: ${medicalInfo.conditions.join(', ')}</span></div>`;
    }
    
    if (medicalInfo.medications && medicalInfo.medications.length > 0) {
        html += `<div class="info-item"><i class="fas fa-pills"></i><span>Medications: ${medicalInfo.medications.join(', ')}</span></div>`;
    }
    
    if (medicalInfo.allergies && medicalInfo.allergies.length > 0) {
        html += `<div class="info-item"><i class="fas fa-exclamation-triangle"></i><span>Allergies: ${medicalInfo.allergies.join(', ')}</span></div>`;
    }
    
    if (medicalInfo.special_needs) {
        html += `<div class="info-item"><i class="fas fa-heart"></i><span>Special Needs: ${medicalInfo.special_needs}</span></div>`;
    }
    
    html += '</div>';
    return html;
}

// Handle drag and drop for file uploads
function setupDragAndDrop(element, callback) {
    element.addEventListener('dragover', (e) => {
        e.preventDefault();
        element.classList.add('dragover');
    });
    
    element.addEventListener('dragleave', (e) => {
        e.preventDefault();
        element.classList.remove('dragover');
    });
    
    element.addEventListener('drop', (e) => {
        e.preventDefault();
        element.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        const validFiles = files.filter(isValidImageFile);
        
        if (validFiles.length !== files.length) {
            showToast('Some files were ignored. Only image files are allowed.', 'warning');
        }
        
        if (validFiles.length > 0) {
            callback(validFiles);
        }
    });
}

// Error handler
function handleError(error, customMessage = null) {
    console.error('Error:', error);
    const message = customMessage || error.message || 'An unexpected error occurred';
    showToast(message, 'error');
    hideLoading();
}
