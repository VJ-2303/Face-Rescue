// Students management functionality
class StudentsManager {
    constructor() {
        this.students = [];
        this.filteredStudents = [];
        this.initializeElements();
    }
    
    initializeElements() {
        const refreshBtn = document.getElementById('refresh-students');
        const searchInput = document.getElementById('students-search');
        
        refreshBtn.addEventListener('click', () => this.loadStudents());
        
        // Debounced search
        searchInput.addEventListener('input', debounce((e) => {
            this.filterStudents(e.target.value);
        }, 300));
        
        // Load students when page becomes active
        document.addEventListener('DOMContentLoaded', () => {
            if (document.getElementById('students-page').classList.contains('active')) {
                this.loadStudents();
            }
        });
    }
    
    async loadStudents() {
        try {
            showLoading('Loading students...');
            
            const students = await api.getStudents();
            this.students = students;
            this.filteredStudents = students;
            
            this.renderStudentsList();
            hideLoading();
            
            showToast(`Loaded ${students.length} students`, 'success');
            
        } catch (error) {
            hideLoading();
            handleError(error, 'Failed to load students');
        }
    }
    
    filterStudents(searchTerm) {
        if (!searchTerm.trim()) {
            this.filteredStudents = this.students;
        } else {
            const term = searchTerm.toLowerCase();
            this.filteredStudents = this.students.filter(student => 
                student.name.toLowerCase().includes(term) ||
                (student.school && student.school.toLowerCase().includes(term)) ||
                (student.emergency_contact.guardian_name.toLowerCase().includes(term)) ||
                (student.emergency_contact.phone.includes(term))
            );
        }
        
        this.renderStudentsList();
    }
    
    renderStudentsList() {
        const container = document.getElementById('students-list');
        
        if (this.filteredStudents.length === 0) {
            container.innerHTML = `
                <div style=\"text-align: center; padding: 3rem; grid-column: 1 / -1;\">
                    <i class=\"fas fa-users\" style=\"font-size: 4rem; color: #ccc; margin-bottom: 1rem;\"></i>
                    <h3 style=\"color: #666; margin-bottom: 0.5rem;\">No Students Found</h3>
                    <p style=\"color: #999;\">
                        ${this.students.length === 0 ? 
                            'No students have been registered yet.' : 
                            'No students match your search criteria.'
                        }
                    </p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.filteredStudents.map(student => 
            this.generateStudentListCard(student)
        ).join('');
    }
    
    generateStudentListCard(student) {
        const initials = getInitials(student.name);
        const formattedDate = formatDate(student.created_at);
        
        return `
            <div class=\"student-list-card\" onclick=\"studentsManager.showStudentDetails('${student.id}')\">
                <div class=\"student-card-header\">
                    <div class=\"student-card-avatar\">${initials}</div>
                    <div class=\"student-card-info\">
                        <h4>${student.name}</h4>
                        <p>${student.school || 'School not specified'}</p>
                    </div>
                </div>
                
                <div class=\"student-card-details\">
                    <div><i class=\"fas fa-birthday-cake\"></i> Age: ${student.age || 'N/A'}</div>
                    <div><i class=\"fas fa-venus-mars\"></i> ${student.gender || 'N/A'}</div>
                    <div><i class=\"fas fa-user\"></i> ${student.emergency_contact.guardian_name}</div>
                    <div><i class=\"fas fa-phone\"></i> ${formatPhoneNumber(student.emergency_contact.phone)}</div>
                    <div><i class=\"fas fa-calendar\"></i> Registered: ${formattedDate}</div>
                    <div><i class=\"fas fa-graduation-cap\"></i> Grade: ${student.grade || 'N/A'}</div>
                </div>
                
                <div class=\"student-card-actions\">
                    <button class=\"btn btn-small btn-primary\" onclick=\"event.stopPropagation(); studentsManager.showStudentDetails('${student.id}')\">
                        <i class=\"fas fa-eye\"></i>
                        View
                    </button>
                    <button class=\"btn btn-small btn-secondary\" onclick=\"event.stopPropagation(); studentsManager.editStudent('${student.id}')\">
                        <i class=\"fas fa-edit\"></i>
                        Edit
                    </button>
                    <button class=\"btn btn-small\" style=\"background: #F44336; color: white;\" onclick=\"event.stopPropagation(); studentsManager.deleteStudent('${student.id}', '${student.name}')\">
                        <i class=\"fas fa-trash\"></i>
                        Delete
                    </button>
                </div>
            </div>
        `;
    }
    
    async showStudentDetails(studentId) {
        try {
            showLoading('Loading student details...');
            
            const student = await api.getStudent(studentId);
            this.displayStudentModal(student);
            
            hideLoading();
            
        } catch (error) {
            hideLoading();
            handleError(error, 'Failed to load student details');
        }
    }
    
    displayStudentModal(student) {
        const initials = getInitials(student.name);
        const formattedDate = formatDate(student.created_at);
        
        const content = `
            <div class=\"student-details\">
                <div class=\"student-header\" style=\"display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid #eee;\">
                    <div class=\"student-avatar\" style=\"width: 80px; height: 80px; font-size: 2rem;\">${initials}</div>
                    <div>
                        <h2 style=\"margin: 0 0 0.5rem 0;\">${student.name}</h2>
                        <p style=\"margin: 0; color: #666;\">Student ID: ${student.id}</p>
                        <p style=\"margin: 0; color: #666;\">Registered: ${formattedDate}</p>
                    </div>
                </div>
                
                <div class=\"student-info-sections\">
                    <!-- Basic Information -->
                    <div class=\"info-section\">
                        <h4 style=\"display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;\">
                            <i class=\"fas fa-user\"></i>
                            Basic Information
                        </h4>
                        <div class=\"info-grid\">
                            ${student.age ? `<div class=\"info-item\"><i class=\"fas fa-birthday-cake\"></i><span>Age: ${student.age}</span></div>` : ''}
                            ${student.gender ? `<div class=\"info-item\"><i class=\"fas fa-venus-mars\"></i><span>Gender: ${student.gender}</span></div>` : ''}
                            ${student.school ? `<div class=\"info-item\"><i class=\"fas fa-school\"></i><span>School: ${student.school}</span></div>` : ''}
                            ${student.grade ? `<div class=\"info-item\"><i class=\"fas fa-graduation-cap\"></i><span>Grade: ${student.grade}</span></div>` : ''}
                        </div>
                    </div>
                    
                    <!-- Emergency Contact -->
                    <div class=\"info-section\">
                        <h4 style=\"display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;\">
                            <i class=\"fas fa-phone\"></i>
                            Emergency Contact
                        </h4>
                        <div class=\"info-grid\">
                            <div class=\"info-item\">
                                <i class=\"fas fa-user\"></i>
                                <span>${student.emergency_contact.guardian_name} (${student.emergency_contact.relationship})</span>
                            </div>
                            <div class=\"info-item\">
                                <i class=\"fas fa-phone\"></i>
                                <span>${formatPhoneNumber(student.emergency_contact.phone)}</span>
                            </div>
                            ${student.emergency_contact.alternate_phone ? `
                                <div class=\"info-item\">
                                    <i class=\"fas fa-phone-alt\"></i>
                                    <span>${formatPhoneNumber(student.emergency_contact.alternate_phone)}</span>
                                </div>
                            ` : ''}
                            ${student.emergency_contact.address ? `
                                <div class=\"info-item\">
                                    <i class=\"fas fa-map-marker-alt\"></i>
                                    <span>${student.emergency_contact.address}</span>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                    
                    ${student.medical_info ? this.generateMedicalInfoSection(student.medical_info) : ''}
                    
                    ${student.notes ? `
                        <div class=\"info-section\">
                            <h4 style=\"display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;\">
                                <i class=\"fas fa-sticky-note\"></i>
                                Additional Notes
                            </h4>
                            <p style=\"background: #f8f9fa; padding: 1rem; border-radius: 6px; margin: 0;\">${student.notes}</p>
                        </div>
                    ` : ''}
                </div>
                
                <div class=\"modal-actions\" style=\"margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; display: flex; gap: 1rem; justify-content: center;\">
                    <a href=\"tel:${student.emergency_contact.phone}\" class=\"btn btn-success\">
                        <i class=\"fas fa-phone\"></i>
                        Call Guardian
                    </a>
                    <a href=\"sms:${student.emergency_contact.phone}\" class=\"btn btn-primary\">
                        <i class=\"fas fa-sms\"></i>
                        Send SMS
                    </a>
                    <button class=\"btn btn-secondary\" onclick=\"studentsManager.editStudent('${student.id}')\">
                        <i class=\"fas fa-edit\"></i>
                        Edit Student
                    </button>
                </div>
            </div>
        `;
        
        showModal('student-modal', content);
    }
    
    generateMedicalInfoSection(medicalInfo) {
        let html = `
            <div class=\"info-section\">
                <h4 style=\"display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;\">
                    <i class=\"fas fa-heartbeat\"></i>
                    Medical Information
                </h4>
                <div class=\"info-grid\">
        `;
        
        if (medicalInfo.conditions && medicalInfo.conditions.length > 0) {
            html += `
                <div class=\"info-item\">
                    <i class=\"fas fa-heartbeat\"></i>
                    <span>Conditions: ${medicalInfo.conditions.join(', ')}</span>
                </div>
            `;
        }
        
        if (medicalInfo.medications && medicalInfo.medications.length > 0) {
            html += `
                <div class=\"info-item\">
                    <i class=\"fas fa-pills\"></i>
                    <span>Medications: ${medicalInfo.medications.join(', ')}</span>
                </div>
            `;
        }
        
        if (medicalInfo.allergies && medicalInfo.allergies.length > 0) {
            html += `
                <div class=\"info-item\">
                    <i class=\"fas fa-exclamation-triangle\"></i>
                    <span>Allergies: ${medicalInfo.allergies.join(', ')}</span>
                </div>
            `;
        }
        
        if (medicalInfo.special_needs) {
            html += `
                <div class=\"info-item\" style=\"grid-column: 1 / -1;\">
                    <i class=\"fas fa-heart\"></i>
                    <span>Special Needs: ${medicalInfo.special_needs}</span>
                </div>
            `;
        }
        
        html += `
                </div>
            </div>
        `;
        
        return html;
    }
    
    editStudent(studentId) {
        // For now, show a message that editing is not implemented
        showToast('Student editing feature coming soon!', 'warning');
        hideModal('student-modal');
        
        // TODO: Implement student editing functionality
        // This would involve:
        // 1. Loading student data into the registration form
        // 2. Switching to register page
        // 3. Pre-filling form fields
        // 4. Changing submit behavior to update instead of create
    }
    
    async deleteStudent(studentId, studentName) {
        const confirmed = confirm(`Are you sure you want to delete ${studentName}? This action cannot be undone.`);
        
        if (confirmed) {
            try {
                showLoading('Deleting student...');
                
                const result = await api.deleteStudent(studentId);
                
                if (result.success) {
                    showToast(`${studentName} has been deleted successfully`, 'success');
                    this.loadStudents(); // Reload the list
                    hideModal('student-modal');
                } else {
                    showToast('Failed to delete student', 'error');
                }
                
                hideLoading();
                
            } catch (error) {
                hideLoading();
                handleError(error, 'Failed to delete student');
            }
        }
    }
    
    // Method to be called when students page becomes active
    onPageActivated() {
        if (this.students.length === 0) {
            this.loadStudents();
        }
    }
}

// Global students manager
const studentsManager = new StudentsManager();

// Add CSS for student details
const style = document.createElement('style');
style.textContent = `
    .student-details .info-section {
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .student-details .info-section:last-child {
        border-bottom: none;
        margin-bottom: 0;
    }
    
    .student-details .info-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.75rem;
    }
    
    .student-details .info-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: #f8f9fa;
        border-radius: 6px;
    }
    
    .student-details .info-item i {
        color: #667eea;
        width: 20px;
        flex-shrink: 0;
    }
    
    @media (min-width: 768px) {
        .student-details .info-grid {
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }
    }
`;
document.head.appendChild(style);
