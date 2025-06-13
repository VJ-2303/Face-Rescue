# Face Rescue - Missing Children Identification System

## 🎯 Project Overview

Face Rescue is a comprehensive web application designed to help identify missing or special needs children using advanced face recognition technology. The system allows administrators to register students with multiple photos and enables quick identification through face scanning.

## 🔧 Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: MongoDB Atlas
- **Face Recognition**: InsightFace with ONNX Runtime
- **Image Processing**: OpenCV, Pillow
- **Vector Similarity**: NumPy with cosine similarity
- **Authentication**: Python-JOSE (JWT tokens)

### Frontend
- **Framework**: Vanilla JavaScript (ES6+)
- **Styling**: CSS3 with modern features
- **Icons**: Font Awesome 6
- **Fonts**: Google Fonts (Inter)
- **Camera API**: WebRTC MediaDevices

## 📁 Project Structure

```
Face-Rescue/
├── .env                          # Environment variables
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── SETUP.md                      # Setup instructions
│
├── backend/                      # FastAPI Backend
│   ├── main.py                   # Application entry point
│   ├── db/
│   │   └── mongodb.py            # Database connection & operations
│   ├── models/
│   │   └── student.py            # Pydantic data models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── register.py           # Student registration endpoints
│   │   └── search.py             # Face search endpoints
│   └── services/
│       ├── face_engine.py        # Face detection & embedding extraction
│       └── matcher.py            # Face matching algorithms
│
└── frontend/                     # Vanilla JS Frontend
    ├── index.html                # Main HTML file
    ├── css/
    │   └── styles.css            # Modern responsive styles
    ├── js/
    │   ├── api.js                # API communication layer
    │   ├── app.js                # Main application controller
    │   ├── camera.js             # Camera management
    │   ├── dashboard.js          # Dashboard functionality
    │   ├── register.js           # Student registration
    │   ├── search.js             # Face search interface
    │   ├── students.js           # Student management
    │   └── utils.js              # Utility functions
    └── assets/
        └── images/               # Static images
```

## 🚀 Features

### Core Functionality
- **Student Registration**: Register students with 3-6 face photos
- **Face Recognition**: Identify students using live camera or uploaded photos
- **Emergency Contacts**: Store and quickly access guardian information
- **Medical Information**: Record medical conditions, allergies, and special needs
- **Search History**: Track all search attempts with timestamps and results

### Advanced Features
- **Batch Search**: Find multiple similar students when exact match isn't found
- **Quality Validation**: Ensure uploaded photos meet quality standards
- **Real-time Camera**: Use device camera for live photo capture
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Dark/Light Theme**: Modern UI with professional appearance

### Security Features
- **Data Privacy**: Only face embeddings stored, not actual photos
- **Input Validation**: Comprehensive validation on both frontend and backend
- **Error Handling**: Graceful error handling with user-friendly messages
- **Rate Limiting**: API protection against abuse

## 📊 API Endpoints

### Student Registration
- `POST /api/register_student` - Register new student with photos
- `GET /api/students` - Get list of all students
- `GET /api/student/{id}` - Get specific student details
- `DELETE /api/student/{id}` - Soft delete student
- `PUT /api/student/{id}/reactivate` - Reactivate deleted student

### Face Search
- `POST /api/search_face` - Search for student by face photo
- `POST /api/search_face/batch` - Get top K similar students
- `GET /api/search_logs` - Get search history
- `GET /api/search_stats` - Get search statistics

### System
- `GET /` - API root endpoint
- `GET /health` - Health check endpoint

## 🔍 How It Works

### Face Recognition Pipeline

1. **Photo Upload/Capture**
   - User uploads or captures photo via camera
   - Image validation and quality checks
   - Automatic resizing and optimization

2. **Face Detection & Extraction**
   - InsightFace detects faces in the image
   - Validates single face presence
   - Extracts face region for processing

3. **Embedding Generation**
   - Generates 512-dimensional face embedding
   - Normalizes embedding vector
   - Stores in MongoDB for future comparisons

4. **Similarity Matching**
   - Compares query embedding with stored embeddings
   - Uses cosine similarity for matching
   - Applies confidence threshold (default: 0.6)

5. **Result Processing**
   - Returns best match above threshold
   - Provides confidence score and match quality
   - Logs search attempt for analytics

### Database Schema

#### Students Collection
```json
{
  "_id": "ObjectId",
  "name": "Student Name",
  "age": 12,
  "gender": "Male/Female/Other",
  "school": "School Name",
  "grade": "Grade/Class",
  "emergency_contact": {
    "guardian_name": "Guardian Name",
    "phone": "+1234567890",
    "alternate_phone": "+0987654321",
    "relationship": "Parent/Guardian/Relative",
    "address": "Full Address"
  },
  "medical_info": {
    "conditions": ["condition1", "condition2"],
    "medications": ["med1", "med2"],
    "allergies": ["allergy1", "allergy2"],
    "special_needs": "Special care instructions"
  },
  "notes": "Additional notes",
  "embeddings": [[512 float values], ...],
  "created_at": "ISO Date",
  "updated_at": "ISO Date",
  "is_active": true
}
```

#### Search Logs Collection
```json
{
  "_id": "ObjectId",
  "search_timestamp": "ISO Date",
  "matched_student_id": "ObjectId or null",
  "confidence": 0.85,
  "ip_address": "User IP",
  "user_agent": "Browser info"
}
```

## 🎨 Frontend Architecture

### Component Structure
- **App Controller**: Main application state management
- **Page Manager**: Handles navigation between pages
- **API Layer**: Centralized API communication
- **Camera Manager**: WebRTC camera operations
- **Search Manager**: Face search functionality
- **Registration Manager**: Student registration workflow
- **Students Manager**: Student list and details
- **Dashboard Manager**: Statistics and analytics

### UI Components
- **Navigation Bar**: Responsive navigation with active states
- **Photo Upload**: Drag & drop with preview
- **Camera Interface**: Live video with face guide overlay
- **Search Results**: Detailed student information display
- **Modal Windows**: Student details and confirmations
- **Toast Notifications**: User feedback system
- **Loading States**: Progress indicators for async operations

## 🔧 Configuration

### Environment Variables
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database
DB_NAME=face_recognition
JWT_SECRET=your-secret-key
PORT=8081
```

### Face Recognition Settings
- **Similarity Threshold**: 0.6 (60% confidence required)
- **Image Size Limit**: 10MB per photo
- **Supported Formats**: JPEG, PNG, WebP
- **Face Detection Model**: Buffalo_L (InsightFace)
- **Embedding Dimension**: 512

## 📱 User Interface

### Pages Overview

1. **Search Page** (Default)
   - Camera interface with live preview
   - Photo upload with drag & drop
   - Search results with emergency contacts
   - Batch search for similar students

2. **Register Page**
   - Student information form
   - Emergency contact details
   - Medical information (optional)
   - Photo upload slots (3-6 photos)
   - Form validation and submission

3. **Students Page**
   - List of all registered students
   - Search and filter functionality
   - Student cards with quick info
   - View, edit, and delete actions

4. **Dashboard Page**
   - System statistics
   - Recent search activity
   - Success rate analytics
   - Real-time updates

### Mobile Responsiveness
- Responsive grid layouts
- Touch-friendly interface
- Mobile camera access
- Optimized for small screens
- Progressive web app ready

## 🔐 Security Considerations

### Data Privacy
- **No Image Storage**: Only face embeddings stored
- **Secure Transmission**: HTTPS for all communications
- **Input Sanitization**: Validation on all user inputs
- **Error Masking**: Generic error messages to prevent information leakage

### Access Control
- **Admin Authentication**: Protected registration endpoints
- **Rate Limiting**: Prevent API abuse
- **CORS Configuration**: Restricted to allowed origins
- **Input Validation**: Server-side validation for all requests

## 🚀 Deployment

### Prerequisites
- Python 3.8+
- MongoDB Atlas account
- Modern web browser
- Camera access (for live capture)

### Local Development
1. Clone the repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Configure environment variables in `.env`
4. Start backend: `cd backend && python main.py`
5. Start frontend: `cd frontend && python -m http.server 3000`
6. Access application at `http://localhost:3000`

### Production Deployment
- **Backend**: Deploy to cloud platforms (AWS, Heroku, DigitalOcean)
- **Frontend**: Serve via CDN or static hosting (Netlify, Vercel)
- **Database**: Use MongoDB Atlas for managed database
- **SSL**: Enable HTTPS for secure communication

## 📈 Performance Optimization

### Frontend Optimizations
- Lazy loading of JavaScript modules
- Image compression and resizing
- Efficient DOM manipulation
- Debounced search functionality
- Browser caching strategies

### Backend Optimizations
- Async/await for database operations
- Face embedding caching
- Efficient vector similarity calculations
- Connection pooling for MongoDB
- Request/response compression

## 🔧 Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Check browser permissions
   - Ensure HTTPS connection
   - Verify camera hardware

2. **Face Detection Fails**
   - Ensure good lighting
   - Single face in photo
   - Clear, unobstructed view

3. **API Connection Issues**
   - Check network connectivity
   - Verify backend server status
   - Review CORS configuration

4. **MongoDB Connection Errors**
   - Validate connection string
   - Check network restrictions
   - Verify credentials

## 🔮 Future Enhancements

### Planned Features
- **Real-time Notifications**: Push notifications for matches
- **Advanced Analytics**: Detailed reporting dashboard
- **Multi-language Support**: Internationalization
- **Mobile App**: Native iOS/Android applications
- **QR Code Backup**: Alternative identification method
- **Advanced Search**: Filter by age, school, etc.
- **Batch Import**: CSV import for multiple students
- **API Documentation**: Interactive Swagger UI

### Technical Improvements
- **GPU Acceleration**: CUDA support for faster processing
- **Vector Database**: Migrate to specialized vector DB (Pinecone, Weaviate)
- **Microservices**: Split into smaller, focused services
- **Kubernetes**: Container orchestration for scaling
- **Real-time Updates**: WebSocket for live updates
- **Advanced AI**: Improved face recognition models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For questions or support, please contact:
- **Technical Issues**: Create an issue on GitHub
- **Feature Requests**: Use the issue tracker
- **Security Concerns**: Email security@faceescape.com

---

**Built with ❤️ for child safety and security**
