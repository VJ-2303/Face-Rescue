// Face Recognition App - Main JavaScript
class FaceRecognitionApp {
    constructor() {
        this.capturedPhotos = [];
        this.webcamStream = null;
        this.searchWebcamStream = null;
        this.currentSection = 'welcome';
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.showSection('welcome');
    }

    setupEventListeners() {
        // Navigation
        document.getElementById('navRegister').addEventListener('click', () => this.showSection('registration'));
        document.getElementById('navSearch').addEventListener('click', () => this.showSection('search'));
        document.getElementById('navList').addEventListener('click', () => this.showSection('list'));
        
        // Welcome section
        document.getElementById('startRegister').addEventListener('click', () => this.showSection('registration'));
        document.getElementById('startSearch').addEventListener('click', () => this.showSection('search'));
        
        // Registration section
        document.getElementById('startCamera').addEventListener('click', () => this.startWebcam());
        document.getElementById('capturePhoto').addEventListener('click', () => this.capturePhoto());
        document.getElementById('uploadButton').addEventListener('click', () => document.getElementById('fileUpload').click());
        document.getElementById('fileUpload').addEventListener('change', (e) => this.handleFileUpload(e));
        document.getElementById('registrationForm').addEventListener('submit', (e) => this.handleRegistration(e));
        
        // Search section
        document.getElementById('startSearchCamera').addEventListener('click', () => this.startSearchWebcam());
        document.getElementById('searchFace').addEventListener('click', () => this.performFaceSearch());
        document.getElementById('searchUploadButton').addEventListener('click', () => document.getElementById('searchFileUpload').click());
        document.getElementById('searchFileUpload').addEventListener('change', (e) => this.handleSearchFileUpload(e));
        
        // List section
        document.getElementById('refreshList').addEventListener('click', () => this.loadStudentsList());
        
        // Modals
        document.getElementById('closeSuccess').addEventListener('click', () => this.hideModal('successModal'));
        document.getElementById('closeError').addEventListener('click', () => this.hideModal('errorModal'));
        
        // Form validation
        this.setupFormValidation();
    }

    setupFormValidation() {
        const requiredFields = ['studentName', 'guardianName', 'relationship', 'primaryPhone', 'address'];
        
        requiredFields.forEach(fieldId => {
            document.getElementById(fieldId).addEventListener('input', () => this.validateForm());
        });
        
        // Photo count validation
        this.validateForm();
    }

    validateForm() {
        const requiredFields = ['studentName', 'guardianName', 'relationship', 'primaryPhone', 'address'];
        const submitBtn = document.getElementById('submitRegistration');
        
        let isValid = true;
        
        // Check required fields
        for (let fieldId of requiredFields) {
            const field = document.getElementById(fieldId);
            if (!field.value.trim()) {
                isValid = false;
                break;
            }
        }
        
        // Check photo count
        if (this.capturedPhotos.length < 3) {
            isValid = false;
        }
        
        submitBtn.disabled = !isValid;
        
        if (isValid) {
            submitBtn.classList.remove('opacity-50');
        } else {
            submitBtn.classList.add('opacity-50');
        }
    }

    showSection(section) {
        // Hide all sections
        document.getElementById('welcomeSection').classList.add('hidden');
        document.getElementById('registrationSection').classList.add('hidden');
        document.getElementById('searchSection').classList.add('hidden');
        document.getElementById('listSection').classList.add('hidden');
        
        // Show selected section
        document.getElementById(section + 'Section').classList.remove('hidden');
        this.currentSection = section;
        
        // Stop any active streams when switching sections
        this.stopAllStreams();
        
        // Load data for specific sections
        if (section === 'list') {
            this.loadStudentsList();
        }
    }

    stopAllStreams() {
        if (this.webcamStream) {
            this.webcamStream.getTracks().forEach(track => track.stop());
            this.webcamStream = null;
        }
        if (this.searchWebcamStream) {
            this.searchWebcamStream.getTracks().forEach(track => track.stop());
            this.searchWebcamStream = null;
        }
        
        // Reset button states
        document.getElementById('capturePhoto').disabled = true;
        document.getElementById('searchFace').disabled = true;
    }

    async startWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            const video = document.getElementById('webcam');
            video.srcObject = stream;
            this.webcamStream = stream;
            
            document.getElementById('capturePhoto').disabled = false;
            
            this.showSuccess('Camera started successfully!');
        } catch (error) {
            console.error('Error starting webcam:', error);
            this.showError('Failed to start camera. Please check camera permissions.');
        }
    }

    async startSearchWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            const video = document.getElementById('searchWebcam');
            video.srcObject = stream;
            this.searchWebcamStream = stream;
            
            document.getElementById('searchFace').disabled = false;
            
            this.showSuccess('Search camera started successfully!');
        } catch (error) {
            console.error('Error starting search webcam:', error);
            this.showError('Failed to start camera. Please check camera permissions.');
        }
    }

    capturePhoto() {
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        context.drawImage(video, 0, 0);
        
        // Show capture overlay animation
        const overlay = document.getElementById('captureOverlay');
        overlay.classList.add('active');
        setTimeout(() => overlay.classList.remove('active'), 1000);
        
        // Convert to blob
        canvas.toBlob((blob) => {
            if (blob) {
                this.addCapturedPhoto(blob);
            }
        }, 'image/jpeg', 0.8);
    }

    addCapturedPhoto(blob) {
        if (this.capturedPhotos.length >= 10) {
            this.showError('Maximum 10 photos allowed.');
            return;
        }
        
        this.capturedPhotos.push(blob);
        this.updatePhotosDisplay();
        this.validateForm();
        
        this.showSuccess(`Photo ${this.capturedPhotos.length} captured successfully!`);
    }

    updatePhotosDisplay() {
        const container = document.getElementById('capturedPhotos');
        container.innerHTML = '';
        
        this.capturedPhotos.forEach((blob, index) => {
            const div = document.createElement('div');
            div.className = 'relative';
            
            const img = document.createElement('img');
            img.src = URL.createObjectURL(blob);
            img.className = 'w-full h-20 object-cover rounded border';
            
            const deleteBtn = document.createElement('button');
            deleteBtn.innerHTML = '<i class="fas fa-times"></i>';
            deleteBtn.className = 'absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 text-xs hover:bg-red-600';
            deleteBtn.onclick = () => this.removePhoto(index);
            
            div.appendChild(img);
            div.appendChild(deleteBtn);
            container.appendChild(div);
        });
        
        if (this.capturedPhotos.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 col-span-2 mt-8">
                    <i class="fas fa-images text-4xl mb-2"></i>
                    <div>Captured photos will appear here</div>
                    <div class="text-sm">Minimum 3 photos required</div>
                </div>
            `;
        }
    }

    removePhoto(index) {
        this.capturedPhotos.splice(index, 1);
        this.updatePhotosDisplay();
        this.validateForm();
    }

    handleFileUpload(event) {
        const files = Array.from(event.target.files);
        
        if (files.length + this.capturedPhotos.length > 10) {
            this.showError('Maximum 10 photos allowed in total.');
            return;
        }
        
        files.forEach(file => {
            if (file.type.startsWith('image/')) {
                this.addCapturedPhoto(file);
            }
        });
        
        event.target.value = ''; // Reset input
    }

    async handleRegistration(event) {
        event.preventDefault();
        
        if (this.capturedPhotos.length < 3) {
            this.showError('Please capture at least 3 photos.');
            return;
        }
        
        this.showLoading('Registering student...');
        
        try {
            const formData = new FormData();
            
            // Add form fields
            formData.append('name', document.getElementById('studentName').value);
            formData.append('age', document.getElementById('studentAge').value || '');
            formData.append('gender', document.getElementById('studentGender').value || '');
            formData.append('school', document.getElementById('studentSchool').value || '');
            formData.append('class_grade', document.getElementById('studentClass').value || '');
            formData.append('special_needs', document.getElementById('specialNeeds').value || '');
            formData.append('guardian_name', document.getElementById('guardianName').value);
            formData.append('relationship', document.getElementById('relationship').value);
            formData.append('phone', document.getElementById('primaryPhone').value);
            formData.append('alternate_phone', document.getElementById('alternatePhone').value || '');
            formData.append('address', document.getElementById('address').value);
            formData.append('email', document.getElementById('email').value || '');
            formData.append('additional_info', document.getElementById('additionalInfo').value || '');
            
            // Add images
            this.capturedPhotos.forEach((blob, index) => {
                formData.append('images', blob, `photo_${index}.jpg`);
            });
            
            const response = await fetch('/api/students/register', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            this.hideModal('loadingModal');
            
            if (response.ok && result.success) {
                this.showSuccess(`Student registered successfully! ${result.embeddings_count} face embeddings created.`);
                this.resetRegistrationForm();
            } else {
                this.showError(result.detail || result.message || 'Registration failed');
            }
            
        } catch (error) {
            this.hideModal('loadingModal');
            console.error('Registration error:', error);
            this.showError('Registration failed. Please check your connection and try again.');
        }
    }

    resetRegistrationForm() {
        document.getElementById('registrationForm').reset();
        this.capturedPhotos = [];
        this.updatePhotosDisplay();
        this.validateForm();
        this.stopAllStreams();
    }

    handleSearchFileUpload(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            this.performFaceSearchWithFile(file);
        }
        event.target.value = ''; // Reset input
    }

    async performFaceSearch() {
        const video = document.getElementById('searchWebcam');
        const canvas = document.getElementById('searchCanvas');
        const context = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        
        canvas.toBlob(async (blob) => {
            if (blob) {
                await this.performFaceSearchWithFile(blob);
            }
        }, 'image/jpeg', 0.8);
    }

    async performFaceSearchWithFile(file) {
        this.showLoading('Searching for matching face...');
        
        try {
            const formData = new FormData();
            formData.append('image', file);
            
            const response = await fetch('/api/search/face', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            this.hideModal('loadingModal');
            this.displaySearchResults(result);
            
        } catch (error) {
            this.hideModal('loadingModal');
            console.error('Search error:', error);
            this.showError('Search failed. Please check your connection and try again.');
        }
    }

    displaySearchResults(result) {
        const container = document.getElementById('searchResults');
        
        if (result.success && result.result && result.result.match_found) {
            const student = result.result.student;
            const confidence = (result.result.confidence * 100).toFixed(1);
            
            container.innerHTML = `
                <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div class="flex items-center mb-3">
                        <i class="fas fa-check-circle text-green-500 text-xl mr-2"></i>
                        <h3 class="text-lg font-bold text-green-800">Match Found!</h3>
                    </div>
                    
                    <div class="space-y-3">
                        <div class="bg-white rounded p-3">
                            <h4 class="font-bold text-lg text-gray-800">${student.name}</h4>
                            <p class="text-sm text-gray-600">Confidence: ${confidence}%</p>
                        </div>
                        
                        <div class="bg-white rounded p-3">
                            <h5 class="font-semibold text-gray-700 mb-2">Student Info:</h5>
                            ${student.age ? `<p><strong>Age:</strong> ${student.age}</p>` : ''}
                            ${student.gender ? `<p><strong>Gender:</strong> ${student.gender}</p>` : ''}
                            ${student.school ? `<p><strong>School:</strong> ${student.school}</p>` : ''}
                            ${student.class_grade ? `<p><strong>Class:</strong> ${student.class_grade}</p>` : ''}
                            ${student.special_needs ? `<p><strong>Special Needs:</strong> ${student.special_needs}</p>` : ''}
                        </div>
                        
                        <div class="bg-red-50 border border-red-200 rounded p-3">
                            <h5 class="font-semibold text-red-700 mb-2">Emergency Contact:</h5>
                            <p><strong>Guardian:</strong> ${student.emergency_contact.guardian_name}</p>
                            <p><strong>Relationship:</strong> ${student.emergency_contact.relationship}</p>
                            <p><strong>Phone:</strong> <a href="tel:${student.emergency_contact.phone}" class="text-blue-600 font-bold">${student.emergency_contact.phone}</a></p>
                            ${student.emergency_contact.alternate_phone ? `<p><strong>Alt Phone:</strong> <a href="tel:${student.emergency_contact.alternate_phone}" class="text-blue-600">${student.emergency_contact.alternate_phone}</a></p>` : ''}
                            <p><strong>Address:</strong> ${student.emergency_contact.address}</p>
                            ${student.emergency_contact.email ? `<p><strong>Email:</strong> <a href="mailto:${student.emergency_contact.email}" class="text-blue-600">${student.emergency_contact.email}</a></p>` : ''}
                        </div>
                        
                        ${student.additional_info ? `
                        <div class="bg-blue-50 border border-blue-200 rounded p-3">
                            <h5 class="font-semibold text-blue-700 mb-2">Additional Info:</h5>
                            <p>${student.additional_info}</p>
                        </div>
                        ` : ''}
                    </div>
                    
                    <div class="mt-4 text-center">
                        <button onclick="window.print()" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition">
                            <i class="fas fa-print mr-2"></i>Print Details
                        </button>
                    </div>
                </div>
            `;
        } else {
            const message = result.message || 'No matching student found.';
            container.innerHTML = `
                <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-center">
                    <i class="fas fa-search text-yellow-500 text-4xl mb-3"></i>
                    <h3 class="text-lg font-bold text-yellow-800 mb-2">No Match Found</h3>
                    <p class="text-gray-600">${message}</p>
                    <p class="text-sm text-gray-500 mt-2">Processing time: ${result.processing_time?.toFixed(2)}s</p>
                </div>
            `;
        }
    }

    async loadStudentsList() {
        this.showLoading('Loading students...');
        
        try {
            const response = await fetch('/api/students/list');
            const students = await response.json();
            
            this.hideModal('loadingModal');
            this.displayStudentsList(students);
            
        } catch (error) {
            this.hideModal('loadingModal');
            console.error('Error loading students:', error);
            this.showError('Failed to load students list.');
        }
    }

    displayStudentsList(students) {
        const container = document.getElementById('studentsList');
        
        if (!students || students.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <i class="fas fa-users text-4xl mb-3"></i>
                    <h3 class="text-lg font-bold mb-2">No Students Registered</h3>
                    <p>Start by registering your first student.</p>
                </div>
            `;
            return;
        }
        
        const studentsHTML = students.map(student => `
            <div class="border border-gray-200 rounded-lg p-4 mb-4 hover:shadow-md transition">
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <h3 class="text-lg font-bold text-gray-800">${student.name}</h3>
                        <div class="text-sm text-gray-600 mt-1">
                            ${student.age ? `Age: ${student.age} | ` : ''}
                            ${student.gender ? `${student.gender} | ` : ''}
                            ${student.school ? `${student.school}` : ''}
                        </div>
                        <div class="text-sm text-gray-500 mt-1">
                            Guardian: ${student.emergency_contact.guardian_name} (${student.emergency_contact.relationship})
                        </div>
                        <div class="text-sm text-gray-500">
                            Phone: ${student.emergency_contact.phone}
                        </div>
                        <div class="text-xs text-gray-400 mt-2">
                            ${student.image_count} face photos | Registered: ${new Date(student.created_at).toLocaleDateString()}
                        </div>
                    </div>
                    <div class="flex space-x-2">
                        <button onclick="app.viewStudent('${student.id}')" class="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600 transition">
                            <i class="fas fa-eye mr-1"></i>View
                        </button>
                        <button onclick="app.deleteStudent('${student.id}', '${student.name}')" class="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600 transition">
                            <i class="fas fa-trash mr-1"></i>Delete
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = `
            <div class="mb-4 text-sm text-gray-600">
                Total Students: ${students.length}
            </div>
            ${studentsHTML}
        `;
    }

    async viewStudent(studentId) {
        this.showLoading('Loading student details...');
        
        try {
            const response = await fetch(`/api/students/${studentId}`);
            const student = await response.json();
            
            this.hideModal('loadingModal');
            
            if (response.ok) {
                this.displayStudentDetails(student);
            } else {
                this.showError('Failed to load student details.');
            }
            
        } catch (error) {
            this.hideModal('loadingModal');
            console.error('Error loading student:', error);
            this.showError('Failed to load student details.');
        }
    }

    displayStudentDetails(student) {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-lg p-6 max-w-md mx-4 max-h-96 overflow-y-auto">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-xl font-bold text-gray-800">${student.name}</h3>
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" class="text-gray-500 hover:text-gray-700">
                        <i class="fas fa-times text-xl"></i>
                    </button>
                </div>
                
                <div class="space-y-3 text-sm">
                    ${student.age ? `<p><strong>Age:</strong> ${student.age}</p>` : ''}
                    ${student.gender ? `<p><strong>Gender:</strong> ${student.gender}</p>` : ''}
                    ${student.school ? `<p><strong>School:</strong> ${student.school}</p>` : ''}
                    ${student.class_grade ? `<p><strong>Class:</strong> ${student.class_grade}</p>` : ''}
                    ${student.special_needs ? `<p><strong>Special Needs:</strong> ${student.special_needs}</p>` : ''}
                    
                    <hr class="my-3">
                    <h4 class="font-semibold text-red-700">Emergency Contact:</h4>
                    <p><strong>Guardian:</strong> ${student.emergency_contact.guardian_name}</p>
                    <p><strong>Relationship:</strong> ${student.emergency_contact.relationship}</p>
                    <p><strong>Phone:</strong> ${student.emergency_contact.phone}</p>
                    ${student.emergency_contact.alternate_phone ? `<p><strong>Alt Phone:</strong> ${student.emergency_contact.alternate_phone}</p>` : ''}
                    <p><strong>Address:</strong> ${student.emergency_contact.address}</p>
                    ${student.emergency_contact.email ? `<p><strong>Email:</strong> ${student.emergency_contact.email}</p>` : ''}
                    
                    ${student.additional_info ? `
                    <hr class="my-3">
                    <h4 class="font-semibold">Additional Info:</h4>
                    <p>${student.additional_info}</p>
                    ` : ''}
                    
                    <hr class="my-3">
                    <p class="text-xs text-gray-500">
                        Face photos: ${student.image_count}<br>
                        Registered: ${new Date(student.created_at).toLocaleDateString()}
                    </p>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }

    async deleteStudent(studentId, studentName) {
        if (!confirm(`Are you sure you want to delete ${studentName}? This action cannot be undone.`)) {
            return;
        }
        
        this.showLoading('Deleting student...');
        
        try {
            const response = await fetch(`/api/students/${studentId}`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            this.hideModal('loadingModal');
            
            if (response.ok && result.success) {
                this.showSuccess('Student deleted successfully.');
                this.loadStudentsList(); // Refresh list
            } else {
                this.showError('Failed to delete student.');
            }
            
        } catch (error) {
            this.hideModal('loadingModal');
            console.error('Error deleting student:', error);
            this.showError('Failed to delete student.');
        }
    }

    showLoading(message) {
        document.getElementById('loadingText').textContent = message;
        document.getElementById('loadingModal').classList.remove('hidden');
    }

    showSuccess(message) {
        document.getElementById('successMessage').textContent = message;
        document.getElementById('successModal').classList.remove('hidden');
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorModal').classList.remove('hidden');
    }

    hideModal(modalId) {
        document.getElementById(modalId).classList.add('hidden');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FaceRecognitionApp();
});

// Handle page unload to stop streams
window.addEventListener('beforeunload', () => {
    if (window.app) {
        window.app.stopAllStreams();
    }
});
