
// JavaScript for Pneumonia Detection System

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const filePreview = document.getElementById('filePreview');
    const previewImage = document.getElementById('previewImage');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // File input change event
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // Drag and drop functionality
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }

    // Form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            displayFilePreview(file);
        }
    }

    function handleDragOver(event) {
        event.preventDefault();
        uploadArea.classList.add('dragover');
    }

    function handleDragLeave(event) {
        event.preventDefault();
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(event) {
        event.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (isValidFileType(file)) {
                fileInput.files = files;
                displayFilePreview(file);
            } else {
                showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF)', 'danger');
            }
        }
    }

    function displayFilePreview(file) {
        if (!isValidFileType(file)) {
            showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF)', 'danger');
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            showAlert('File size must be less than 16MB', 'danger');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            fileName.textContent = `File: ${file.name}`;
            fileSize.textContent = `Size: ${formatFileSize(file.size)}`;

            filePreview.style.display = 'block';
            submitBtn.disabled = false;

            // Add animation
            filePreview.classList.add('fade-in');
        };
        reader.readAsDataURL(file);
    }

    function handleFormSubmit(event) {
        event.preventDefault();

        if (!fileInput.files || fileInput.files.length === 0) {
            showAlert('Please select a file first', 'warning');
            return;
        }

        // Show loading indicator
        loadingIndicator.style.display = 'block';
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';

        // Submit form
        const formData = new FormData(uploadForm);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                return response.text();
            }
            throw new Error('Network response was not ok');
        })
        .then(html => {
            // Replace current page with result
            document.open();
            document.write(html);
            document.close();
        })
        .catch(error) {
            console.error('Error:', error);
            hideLoadingIndicator();
            showAlert('An error occurred while processing your request. Please try again.', 'danger');
        });
    }

    function isValidFileType(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif'];
        return validTypes.includes(file.type);
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function hideLoadingIndicator() {
        loadingIndicator.style.display = 'none';
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-diagnoses me-2"></i>Analyze X-Ray';
    }

    function showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.dynamic-alert');
        existingAlerts.forEach(alert => alert.remove());

        // Create new alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show dynamic-alert`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Insert alert before the upload form
        const container = document.querySelector('.container');
        const uploadSection = document.getElementById('upload-section');
        if (uploadSection) {
            uploadSection.insertBefore(alertDiv, uploadSection.firstChild);
        }

        // Auto-remove alert after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    // Clear file selection
    window.clearFile = function() {
        fileInput.value = '';
        filePreview.style.display = 'none';
        submitBtn.disabled = true;
        uploadArea.classList.remove('dragover');
    };

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading animation to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function() {
            if (!this.disabled && !this.classList.contains('no-loading')) {
                this.classList.add('loading');
            }
        });
    });

    // Initialize tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // API endpoint for programmatic access
    window.predictFromAPI = function(fileData) {
        const formData = new FormData();
        formData.append('file', fileData);

        return fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .catch(error => {
            console.error('API Error:', error);
            return { error: 'Failed to process request' };
        });
    };

    // Add keyboard navigation
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            // Close any open modals or clear file selection
            if (filePreview.style.display === 'block') {
                clearFile();
            }
        }

        if (e.ctrlKey && e.key === 'Enter') {
            // Submit form with Ctrl+Enter
            if (submitBtn && !submitBtn.disabled) {
                uploadForm.submit();
            }
        }
    });

    // Progressive Web App features
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => console.log('SW registered'))
            .catch(error => console.log('SW registration failed'));
    }

    // Add visual feedback for interactive elements
    document.querySelectorAll('.card, .btn, .upload-area').forEach(element => {
        element.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });

        element.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    console.log('Pneumonia Detection System initialized successfully!');
});

// Utility functions
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showNotification('Copied to clipboard!', 'success');
    });
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} position-fixed`;
    notification.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        animation: slideInRight 0.3s ease;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Auto-remove notification
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, 3000);
}

// Export functions for use in other scripts
window.PneumoniaApp = {
    clearFile: window.clearFile,
    predictFromAPI: window.predictFromAPI,
    showNotification: showNotification,
    copyToClipboard: copyToClipboard
};
