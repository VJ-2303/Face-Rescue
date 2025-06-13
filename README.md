# Face Recognition App - Missing/Special Children Identification

A fullstack web application built with React + FastAPI + MongoDB for identifying missing or special children using face recognition technology.

## 🎯 Features

- **Admin Registration**: Register students with multiple face photos
- **Face Search**: Identify children by scanning their face
- **Emergency Info**: Store and retrieve emergency contact details
- **Secure Storage**: Only face embeddings stored, not actual images

## 🔧 Tech Stack

- **Frontend**: React.js, Tailwind CSS
- **Backend**: FastAPI, Python
- **AI**: InsightFace for face embeddings
- **Database**: MongoDB Atlas
- **Vector Search**: NumPy with cosine similarity

## 📁 Project Structure

```
Face-Rescue/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── routers/
│   │   ├── register.py         # Student registration
│   │   └── search.py           # Face search
│   ├── services/
│   │   ├── face_engine.py      # Face embedding logic
│   │   └── matcher.py          # Matching algorithm
│   ├── db/
│   │   └── mongodb.py          # Database connection
│   └── models/
│       └── student.py          # Data models
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── api/
└── requirements.txt
```

## 🚀 Setup Instructions

### Backend Setup
```bash
cd backend
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8081
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/register_student` | POST | Register new student with photos |
| `/search_face` | POST | Search for student by face |
| `/students` | GET | List all registered students |
| `/student/{id}` | DELETE | Remove student record |

## 🔐 Environment Variables

Create a `.env` file with:
```
MONGODB_URI=your_mongodb_connection_string
DB_NAME=face_recognition
JWT_SECRET=your_jwt_secret
PORT=8081
```

## 🎯 Usage

1. **Admin Registration**: Upload 4-5 clear photos of the child
2. **Search**: Capture or upload a photo to identify the child
3. **Results**: Get emergency contact information instantly

## 📋 Development Phases

- ✅ Phase 1: Student Registration System
- ✅ Phase 2: Face Search & Identification
- 🔄 Phase 3: Admin Dashboard (In Progress)

## 🔒 Privacy & Security

- Only face embeddings stored, not images
- HTTPS encryption for all API calls
- Configurable similarity thresholds
- Admin authentication for sensitive operations
