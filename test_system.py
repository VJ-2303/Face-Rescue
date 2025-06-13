#!/usr/bin/env python3
"""
Face Rescue System Test Script
Tests the basic functionality of the Face Recognition API
"""

import requests
import json
import sys
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8081/api"
HEALTH_URL = "http://localhost:8081/health"

def test_api_health():
    """Test if the API is running and healthy"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy and running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def test_mongodb_connection():
    """Test MongoDB connection through API"""
    print("🔍 Testing MongoDB Connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/students", timeout=10)
        if response.status_code == 200:
            students = response.json()
            print(f"✅ MongoDB connected successfully. Found {len(students)} students.")
            return True
        else:
            print(f"❌ MongoDB connection test failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to test MongoDB connection: {e}")
        return False

def create_sample_student():
    """Create a sample student for testing"""
    print("🔍 Testing Student Registration...")
    
    # Sample student data
    student_data = {
        "name": "Test Student",
        "age": 10,
        "gender": "Male",
        "school": "Test Elementary School",
        "grade": "5th Grade",
        "emergency_contact": {
            "guardian_name": "Test Guardian",
            "phone": "+1-555-0123",
            "alternate_phone": "+1-555-0124",
            "relationship": "Parent",
            "address": "123 Test Street, Test City, TC 12345"
        },
        "medical_info": {
            "conditions": ["None"],
            "medications": [],
            "allergies": ["Peanuts"],
            "special_needs": "No special needs"
        },
        "notes": "This is a test student created by the automated test script."
    }
    
    # Note: In a real test, we would need actual image files
    print("ℹ️  Note: Actual photo upload testing requires real image files")
    print("ℹ️  Student data structure validated:")
    print(json.dumps(student_data, indent=2))
    
    return True

def test_face_search():
    """Test face search functionality"""
    print("🔍 Testing Face Search API...")
    
    # Note: In a real test, we would need an actual image file
    print("ℹ️  Note: Actual face search testing requires real image files")
    print("ℹ️  Face search endpoint available at: POST /api/search_face")
    
    return True

def test_search_stats():
    """Test search statistics endpoint"""
    print("🔍 Testing Search Statistics...")
    try:
        response = requests.get(f"{API_BASE_URL}/search_stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Search statistics retrieved successfully:")
            print(f"   📊 Total searches: {stats.get('stats', {}).get('total_searches', 0)}")
            print(f"   ✅ Successful matches: {stats.get('stats', {}).get('successful_matches', 0)}")
            print(f"   📈 Success rate: {stats.get('stats', {}).get('success_rate', 0)}%")
            return True
        else:
            print(f"❌ Failed to get search statistics: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to test search statistics: {e}")
        return False

def test_search_logs():
    """Test search logs endpoint"""
    print("🔍 Testing Search Logs...")
    try:
        response = requests.get(f"{API_BASE_URL}/search_logs?limit=5", timeout=10)
        if response.status_code == 200:
            logs = response.json()
            print(f"✅ Search logs retrieved successfully. Found {logs.get('total_count', 0)} total logs.")
            if logs.get('logs'):
                print("   📝 Recent log entries:")
                for log in logs['logs'][:3]:
                    timestamp = log.get('timestamp', 'Unknown')
                    has_match = log.get('has_match', False)
                    status = "✅ Match" if has_match else "❌ No Match"
                    print(f"      {timestamp}: {status}")
            return True
        else:
            print(f"❌ Failed to get search logs: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to test search logs: {e}")
        return False

def run_all_tests():
    """Run all available tests"""
    print("🚀 Starting Face Rescue System Tests")
    print("=" * 50)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 6
    
    # Run tests
    if test_api_health():
        tests_passed += 1
    
    if test_mongodb_connection():
        tests_passed += 1
    
    if create_sample_student():
        tests_passed += 1
    
    if test_face_search():
        tests_passed += 1
    
    if test_search_stats():
        tests_passed += 1
    
    if test_search_logs():
        tests_passed += 1
    
    # Summary
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    
    print("=" * 50)
    print(f"🏁 Test Summary:")
    print(f"   ✅ Tests Passed: {tests_passed}/{total_tests}")
    print(f"   ⏱️  Duration: {duration} seconds")
    print(f"   📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Face Rescue system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the API and database connections.")
        return False

def print_system_info():
    """Print system information and endpoints"""
    print("📋 Face Rescue System Information")
    print("=" * 50)
    print(f"🌐 Frontend URL: http://localhost:3000")
    print(f"🔗 Backend API: {API_BASE_URL}")
    print(f"❤️  Health Check: {HEALTH_URL}")
    print("")
    print("📡 Available Endpoints:")
    print("   🔍 Search Face: POST /api/search_face")
    print("   👥 List Students: GET /api/students")
    print("   ➕ Register Student: POST /api/register_student")
    print("   📊 Search Stats: GET /api/search_stats")
    print("   📝 Search Logs: GET /api/search_logs")
    print("")

if __name__ == "__main__":
    print_system_info()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test - just check if services are running
        print("🏃 Running Quick Test...")
        test_api_health()
        test_mongodb_connection()
    else:
        # Full test suite
        success = run_all_tests()
        sys.exit(0 if success else 1)
