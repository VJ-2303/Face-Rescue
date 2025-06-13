// API Configuration
const API_BASE_URL = 'http://localhost:8081/api';

class API {
    constructor() {
        this.baseURL = API_BASE_URL;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    ...options.headers,
                }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    async get(endpoint) {
        return this.request(endpoint, {
            method: 'GET'
        });
    }

    async post(endpoint, data, isFormData = false) {
        const options = {
            method: 'POST'
        };

        if (isFormData) {
            options.body = data;
        } else {
            options.headers = {
                'Content-Type': 'application/json'
            };
            options.body = JSON.stringify(data);
        }

        return this.request(endpoint, options);
    }

    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
    }

    async delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }

    // Student Registration
    async registerStudent(studentData, photos) {
        const formData = new FormData();
        formData.append('student_data', JSON.stringify(studentData));
        
        photos.forEach((photo, index) => {
            formData.append('photos', photo, `photo_${index}.jpg`);
        });

        return this.post('/register_student', formData, true);
    }

    // Face Search
    async searchFace(photo) {
        const formData = new FormData();
        formData.append('photo', photo, 'search_photo.jpg');

        return this.post('/search_face', formData, true);
    }

    // Batch Face Search
    async batchSearchFace(photo, topK = 3) {
        const formData = new FormData();
        formData.append('photo', photo, 'search_photo.jpg');

        return this.post(`/search_face/batch?top_k=${topK}`, formData, true);
    }

    // Students Management
    async getStudents() {
        return this.get('/students');
    }

    async getStudent(studentId) {
        return this.get(`/student/${studentId}`);
    }

    async deleteStudent(studentId) {
        return this.delete(`/student/${studentId}`);
    }

    async reactivateStudent(studentId) {
        return this.put(`/student/${studentId}/reactivate`, {});
    }

    // Search Logs and Statistics
    async getSearchLogs(limit = 50, offset = 0) {
        return this.get(`/search_logs?limit=${limit}&offset=${offset}`);
    }

    async getSearchStats() {
        return this.get('/search_stats');
    }

    // Health Check
    async healthCheck() {
        return this.get('/health');
    }
}

// Global API instance
const api = new API();
