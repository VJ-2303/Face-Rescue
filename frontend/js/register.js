// Student registration functionality
class RegistrationManager {
    constructor() {
        this.studentPhotos = [];
        this.maxPhotos = 6;
        this.minPhotos = 3;
        this.initializeElements();
        this.generatePhotoSlots();
    }
    
    initializeElements() {
        const form = document.getElementById('register-form');
        const clearBtn = document.getElementById('clear-form');
        
        form.addEventListener('submit', (e) => this.handleSubmit(e));
        clearBtn.addEventListener('click', () => this.clearForm());
        
        // Add input validation
        this.setupValidation();
    }
    
    generatePhotoSlots() {
        const photosGrid = document.getElementById('photos-grid');
        photosGrid.innerHTML = '';
        
        for (let i = 0; i < this.maxPhotos; i++) {
            const slot = this.createPhotoSlot(i);
            photosGrid.appendChild(slot);
        }
    }
    
    createPhotoSlot(index) {
        const slot = document.createElement('div');
        slot.className = 'photo-slot';
        slot.dataset.index = index;
        
        slot.innerHTML = `
            <i class=\"fas fa-camera\" style=\"font-size: 2rem; color: #ccc; margin-bottom: 0.5rem;\"></i>
            <span style=\"color: #666; font-size: 0.9rem;\">Photo ${index + 1}</span>
            <input type=\"file\" accept=\"image/*\" style=\"display: none;\">
        `;
        
        const input = slot.querySelector('input');
        
        slot.addEventListener('click', () => {
            if (!slot.classList.contains('has-image')) {
                input.click();
            }
        });
        
        input.addEventListener('change', (e) => this.handlePhotoSelect(e, index));
        
        // Drag and drop
        setupDragAndDrop(slot, (files) => {
            if (files.length > 0 && !slot.classList.contains('has-image')) {
                this.addPhoto(files[0], index);
            }
        });
        
        return slot;
    }
    
    handlePhotoSelect(event, index) {
        const file = event.target.files[0];
        if (file && isValidImageFile(file)) {
            this.addPhoto(file, index);
        } else if (file) {
            showToast('Please select a valid image file', 'error');
        }
    }
    
    async addPhoto(file, index) {
        try {
            // Check file size
            if (file.size > 10 * 1024 * 1024) {
                showToast('File size too large. Please select an image under 10MB', 'error');
                return;
            }
            
            showLoading('Processing photo...');
            
            // Resize image
            const resizedFile = await resizeImage(file, 800, 600, 0.8);
            
            // Update the photo slot
            const slot = document.querySelector(`[data-index=\"${index}\"]`);
            const imageUrl = URL.createObjectURL(resizedFile);
            
            slot.innerHTML = `
                <img src=\"${imageUrl}\" alt=\"Student photo ${index + 1}\">
                <button type=\"button\" class=\"remove-photo\" title=\"Remove photo\">
                    <i class=\"fas fa-times\"></i>
                </button>
            `;
            
            slot.classList.add('has-image');
            
            // Add remove functionality
            const removeBtn = slot.querySelector('.remove-photo');
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.removePhoto(index);
            });
            
            // Store the photo
            this.studentPhotos[index] = resizedFile;
            
            hideLoading();
            showToast(`Photo ${index + 1} added successfully`, 'success');
            
        } catch (error) {
            hideLoading();
            handleError(error, 'Error processing photo');
        }
    }
    
    removePhoto(index) {
        const slot = document.querySelector(`[data-index=\"${index}\"]`);
        
        // Reset slot
        slot.innerHTML = `
            <i class=\"fas fa-camera\" style=\"font-size: 2rem; color: #ccc; margin-bottom: 0.5rem;\"></i>
            <span style=\"color: #666; font-size: 0.9rem;\">Photo ${index + 1}</span>
            <input type=\"file\" accept=\"image/*\" style=\"display: none;\">
        `;
        
        slot.classList.remove('has-image');
        
        // Re-add event listeners
        const input = slot.querySelector('input');
        slot.addEventListener('click', () => input.click());
        input.addEventListener('change', (e) => this.handlePhotoSelect(e, index));
        
        // Remove from array
        delete this.studentPhotos[index];
        
        showToast(`Photo ${index + 1} removed`, 'success');
    }
    
    setupValidation() {
        const form = document.getElementById('register-form');
        const inputs = form.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            input.addEventListener('blur', () => this.validateField(input));
            input.addEventListener('input', () => this.clearFieldError(input));
        });
    }
    
    validateField(field) {
        const value = field.value.trim();
        let isValid = true;
        let message = '';
        
        // Required field validation
        if (field.hasAttribute('required') && !value) {
            isValid = false;
            message = 'This field is required';
        }
        
        // Specific field validations
        if (value && field.type === 'email') {
            const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
            if (!emailRegex.test(value)) {
                isValid = false;
                message = 'Please enter a valid email address';
            }
        }
        
        if (value && field.type === 'tel') {
            const phoneRegex = /^[\\d\\s\\-\\(\\)\\+]+$/;
            if (!phoneRegex.test(value) || value.replace(/\\D/g, '').length < 10) {
                isValid = false;
                message = 'Please enter a valid phone number';
            }
        }
        
        if (value && field.name === 'age') {
            const age = parseInt(value);
            if (age < 1 || age > 25) {
                isValid = false;
                message = 'Age must be between 1 and 25';
            }
        }
        
        this.showFieldValidation(field, isValid, message);
        return isValid;
    }
    
    showFieldValidation(field, isValid, message) {
        // Remove existing error
        const existingError = field.parentNode.querySelector('.error-text');
        if (existingError) {
            existingError.remove();
        }
        
        if (!isValid) {
            field.style.borderColor = '#F44336';
            
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-text';
            errorDiv.textContent = message;
            field.parentNode.appendChild(errorDiv);
        } else {
            field.style.borderColor = '#ddd';
        }
    }
    
    clearFieldError(field) {
        field.style.borderColor = '#ddd';
        const existingError = field.parentNode.querySelector('.error-text');
        if (existingError) {
            existingError.remove();
        }
    }
    
    async handleSubmit(event) {
        event.preventDefault();
        
        try {
            // Validate form
            const form = event.target;
            const errors = validateForm(form);
            
            // Validate photos
            const photoCount = this.studentPhotos.filter(photo => photo).length;
            if (photoCount < this.minPhotos) {
                errors.push(`At least ${this.minPhotos} photos are required`);
            }
            
            if (errors.length > 0) {
                showToast(errors[0], 'error');
                return;
            }
            
            showLoading('Registering student...');
            
            // Collect form data
            const formData = new FormData(form);
            const studentData = this.collectStudentData(formData);
            
            // Get photos array (filter out empty slots)
            const photos = this.studentPhotos.filter(photo => photo);
            
            // Submit to API
            const result = await api.registerStudent(studentData, photos);
            
            hideLoading();
            
            if (result.success) {
                showToast(`Student registered successfully! ID: ${result.student_id}`, 'success');
                this.clearForm();
                
                // Show success modal with details
                this.showSuccessModal(result);
            } else {
                showToast('Registration failed', 'error');
            }
            
        } catch (error) {
            hideLoading();
            handleError(error, 'Registration failed. Please check your information and try again.');
        }
    }
    
    collectStudentData(formData) {
        const data = {
            name: formData.get('name'),
            age: formData.get('age') ? parseInt(formData.get('age')) : null,
            gender: formData.get('gender') || null,
            school: formData.get('school') || null,
            grade: formData.get('grade') || null,
            emergency_contact: {
                guardian_name: formData.get('guardian_name'),
                phone: formData.get('phone'),
                alternate_phone: formData.get('alternate_phone') || null,
                relationship: formData.get('relationship'),
                address: formData.get('address') || null
            },
            notes: formData.get('notes') || null
        };
        
        // Medical information
        const conditions = formData.get('conditions');
        const medications = formData.get('medications');
        const allergies = formData.get('allergies');
        const specialNeeds = formData.get('special_needs');
        
        if (conditions || medications || allergies || specialNeeds) {
            data.medical_info = {
                conditions: conditions ? conditions.split(',').map(s => s.trim()).filter(s => s) : [],
                medications: medications ? medications.split(',').map(s => s.trim()).filter(s => s) : [],
                allergies: allergies ? allergies.split(',').map(s => s.trim()).filter(s => s) : [],
                special_needs: specialNeeds || null
            };
        }
        
        return data;
    }
    
    showSuccessModal(result) {
        const content = `
            <div style=\"text-align: center;\">
                <i class=\"fas fa-check-circle\" style=\"font-size: 4rem; color: #4CAF50; margin-bottom: 1rem;\"></i>
                <h3 style=\"margin-bottom: 1rem; color: #333;\">Registration Successful!</h3>
                <div style=\"background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;\">
                    <p><strong>Student ID:</strong> ${result.student_id}</p>
                    <p><strong>Photos Processed:</strong> ${result.photos_processed}</p>
                    <p><strong>Face Embeddings:</strong> ${result.embeddings_count}</p>
                </div>
                <p style=\"color: #666; margin-bottom: 2rem;\">
                    The student has been successfully registered in the system and can now be identified through face recognition.
                </p>
                <button class=\"btn btn-primary\" onclick=\"hideModal('student-modal')\">
                    <i class=\"fas fa-check\"></i>
                    Continue
                </button>
            </div>
        `;
        
        showModal('student-modal', content);
    }
    
    clearForm() {
        const form = document.getElementById('register-form');
        clearForm(form);
        
        // Clear photos
        this.studentPhotos = [];
        this.generatePhotoSlots();
        
        showToast('Form cleared', 'success');
    }
}

// Global registration manager
const registrationManager = new RegistrationManager();
