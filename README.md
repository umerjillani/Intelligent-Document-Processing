# Document Processing System

A comprehensive system for processing and managing documents with a React frontend and Flask backend.

## Tech Stack

### Frontend
- **Framework**: React with JavaScript
- **Requirement**: Node.js

### Backend
- **Framework**: Flask (Python)
- **Requirement**: Python

### Database
- **Type**: MySQL

## Getting Started

### Prerequisites
- Node.js
- Python
- MySQL

### Frontend Setup
1. Clone the repository
   ```
   git clone <repository-url>
   ```
2. Navigate to the Frontend folder
   ```
   cd Frontend
   ```
3. Install dependencies and start the development server
   ```
   npm install
   npm run dev
   ```
4. The frontend server will start on `localhost:5173`

### Backend Setup
1. Navigate to the Backend folder
   ```
   cd Backend
   ```
2. Install required Python packages
   ```
   pip install -r requirements.txt
   ```
3. Start the server
   ```
   python server.py
   ```
4. The backend server will start on `localhost:3000`

### Database Setup
- The database schema is provided in the `Script.sql` file
- Import this schema into your MySQL database

## Usage Guide

### Document Processing Workflow

1. **Uploading Documents**
   - Navigate to the "Document Upload" section
   - Select one or multiple documents for processing
   - Click the "Process" button to begin document processing

2. **Processing**
   - Files are sent to the backend for processing
   - The system will handle each document according to configured rules
   - Wait for the processing to complete

3. **Results**
   - A popup will display processing results upon completion
   - The summary shows successfully processed documents and exceptions
   - You can view detailed results in their respective sections

## Troubleshooting

If you encounter any issues during setup or processing, check the following:
- Ensure all prerequisites are installed correctly
- Verify that both frontend and backend servers are running
- Check database connection settings
- Review logs for detailed error information
