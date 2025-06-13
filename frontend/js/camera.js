// Camera functionality
class CameraManager {
    constructor() {
        this.stream = null;
        this.video = document.getElementById('camera-video');
        this.canvas = document.getElementById('camera-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.isActive = false;
        
        this.initializeElements();
    }
    
    initializeElements() {
        const startBtn = document.getElementById('start-camera');
        const captureBtn = document.getElementById('capture-photo');
        const stopBtn = document.getElementById('stop-camera');
        
        startBtn.addEventListener('click', () => this.startCamera());
        captureBtn.addEventListener('click', () => this.capturePhoto());
        stopBtn.addEventListener('click', () => this.stopCamera());
    }
    
    async startCamera() {
        try {
            showLoading('Starting camera...');
            
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            this.isActive = true;
            
            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            });
            
            this.updateButtons();
            hideLoading();
            showToast('Camera started successfully', 'success');
            
        } catch (error) {
            hideLoading();
            console.error('Camera error:', error);
            
            let message = 'Failed to start camera. ';
            if (error.name === 'NotAllowedError') {
                message += 'Please allow camera access.';
            } else if (error.name === 'NotFoundError') {
                message += 'No camera found.';
            } else {
                message += 'Please check your camera settings.';
            }
            
            showToast(message, 'error');
        }
    }
    
    capturePhoto() {
        if (!this.isActive || !this.video.videoWidth) {
            showToast('Camera is not ready', 'error');
            return null;
        }
        
        // Draw video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert to blob
        return new Promise((resolve) => {
            this.canvas.toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
                    
                    // Show preview
                    const previewSection = document.getElementById('preview-section');
                    const previewImage = document.getElementById('preview-image');
                    
                    previewImage.src = URL.createObjectURL(blob);
                    previewSection.style.display = 'block';
                    
                    // Scroll to preview
                    previewSection.scrollIntoView({ behavior: 'smooth' });
                    
                    showToast('Photo captured successfully', 'success');
                    resolve(file);
                } else {
                    showToast('Failed to capture photo', 'error');
                    resolve(null);
                }
            }, 'image/jpeg', 0.8);
        });
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.video.srcObject = null;
        this.isActive = false;
        this.updateButtons();
        
        showToast('Camera stopped', 'success');
    }
    
    updateButtons() {
        const startBtn = document.getElementById('start-camera');
        const captureBtn = document.getElementById('capture-photo');
        const stopBtn = document.getElementById('stop-camera');
        const overlay = document.querySelector('.camera-overlay');
        
        if (this.isActive) {
            startBtn.style.display = 'none';
            captureBtn.style.display = 'inline-flex';
            stopBtn.style.display = 'inline-flex';
            overlay.style.display = 'none';
        } else {
            startBtn.style.display = 'inline-flex';
            captureBtn.style.display = 'none';
            stopBtn.style.display = 'none';
            overlay.style.display = 'flex';
        }
    }
    
    isRunning() {
        return this.isActive;
    }
}

// Global camera manager
const cameraManager = new CameraManager();
