from flask import Flask, request, jsonify
# from new import process_directory
from flask_cors import CORS
from doc_processor import process_files
import os
import mysql.connector
from dbOps import get_db_connection
import json

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Retrieve database connection info from environment variables
db_config = {
    'host': os.getenv('MYSQL_HOST'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DATABASE')
}

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

conn = mysql.connector.connect(**db_config)
if conn.is_connected():
    print("Connected to MySQL database!")
cursor = conn.cursor()


@app.route('/api/process_docs', methods=['POST'])
def process_docs():
    print(request.files)
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    file_objects = request.files.getlist('files')
    results = process_files(file_objects, os.path.join("Files", "JSONs"))
    return jsonify(results)

@app.route('/api/get_docs', methods=['GET'])
def get_documents():

    conn = get_db_connection()
    sql = """
        SELECT d.id, d.doc_name, b.batch_name, dt.type_name, d.status, d.created_at, d.json
        FROM Documents d
        JOIN Batches b ON d.batch_id = b.id
        JOIN DocumentTypes dt ON d.type_id = dt.id
        ORDER BY d.created_at DESC;
        """

    # Execute query
    cursor.execute(sql)

    # Fetch all rows
    rows = cursor.fetchall()
    # Get column names from the cursor
    column_names = [desc[0] for desc in cursor.description]

    # Convert to list of dictionaries
    documents = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    
    return jsonify(documents)

from flask import jsonify
import mysql.connector

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    # SQL Queries
    queries = {
        "documents_today": """
            SELECT COUNT(*) AS documents_today 
            FROM Documents 
            WHERE DATE(created_at) = CURDATE();
        """,
        
        "processing_rate": """
            SELECT 
                ROUND((COUNT(CASE WHEN status = 'processed' THEN 1 END) / COUNT(*)) * 100, 2) 
                AS processing_rate 
            FROM Documents;
        """,
        
        "pending_exceptions": """
            SELECT COUNT(*) AS pending_exceptions 
            FROM Exceptions 
            WHERE resolved = FALSE;
        """,
        
        "document_type_breakdown": """
            SELECT 
                dt.type_name, 
                COUNT(d.id) AS count,
                ROUND((COUNT(d.id) / (SELECT COUNT(*) FROM Documents)) * 100, 2) AS percentage
            FROM Documents d
            JOIN DocumentTypes dt ON d.type_id = dt.id
            GROUP BY dt.type_name
            ORDER BY count DESC;
        """,
        
        "recent_batches": """
            SELECT 
                b.id,
                b.batch_name AS batch_id,
                COUNT(d.id) AS documents,
                CASE 
                    WHEN EXISTS (
                        SELECT 1 FROM Documents 
                        WHERE batch_id = b.id AND status = 'exception'
                    ) THEN 'Exceptions'
                    WHEN COUNT(CASE WHEN d.status = 'processed' THEN 1 END) = COUNT(d.id) THEN 'Complete'
                    ELSE 'Processing'
                END AS status
            FROM Batches b
            LEFT JOIN Documents d ON b.id = d.batch_id
            GROUP BY b.id, b.batch_name
            ORDER BY b.created_at DESC
            LIMIT 4;
        """,
        
        "recent_exceptions": """
            SELECT 
                d.id AS doc_id,
                dt.type_name AS type,
                e.exception_message AS issue
            FROM Exceptions e
            JOIN Documents d ON e.document_id = d.id
            JOIN DocumentTypes dt ON d.type_id = dt.id
            WHERE e.resolved = FALSE
            ORDER BY e.created_at DESC
            LIMIT 4;
        """
    }

    response_data = {}

    try:
        for key, query in queries.items():
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]
            response_data[key] = data

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

    """
    Response Structure
    {
      "documents_today": [
        {
          "documents_today": 12
        }
      ],
      "processing_rate": [
        {
          "processing_rate": 75.0
        }
      ],
      "pending_exceptions": [
        {
          "pending_exceptions": 3
        }
      ],
      "document_type_breakdown": [
        {
          "type_name": "Claim",
          "count": 50,
          "percentage": 33.33
        },
        {
          "type_name": "Check",
          "count": 45,
          "percentage": 30.0
        },
        {
          "type_name": "Mortgage",
          "count": 30,
          "percentage": 20.0
        },
        {
          "type_name": "Coupon",
          "count": 25,
          "percentage": 16.67
        }
      ],
      "recent_batches": [
        {
          "id": 101,
          "batch_id": "Batch_0420A",
          "documents": 20,
          "status": "Processing"
        },
        {
          "id": 100,
          "batch_id": "Batch_0419B",
          "documents": 15,
          "status": "Exceptions"
        },
        {
          "id": 99,
          "batch_id": "Batch_0418C",
          "documents": 25,
          "status": "Complete"
        },
        {
          "id": 98,
          "batch_id": "Batch_0417D",
          "documents": 30,
          "status": "Processing"
        }
      ],
      "recent_exceptions": [
        {
          "doc_id": 89,
          "type": "Mortgage",
          "issue": "Missing address field"
        },
        {
          "doc_id": 84,
          "type": "Claim",
          "issue": "Invalid claim number format"
        },
        {
          "doc_id": 82,
          "type": "Coupon",
          "issue": "Amount exceeds limit"
        },
        {
          "doc_id": 80,
          "type": "Check",
          "issue": "Loan number missing"
        }
      ]
    }

    """
    


@app.route('/api/exceptions', methods=['GET'])
def get_exceptions():  
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT
            d.id AS documentId,
            d.doc_name AS fileName,
            dt.type_name AS type,
            e.exception_type AS exceptionType,
            b.batch_name AS batchId,
            DATE_FORMAT(e.created_at, '%Y-%m-%d') AS dateIdentified,
            d.json AS json
        FROM Documents d
        JOIN DocumentTypes dt ON d.type_id = dt.id
        JOIN Batches b ON d.batch_id = b.id
        JOIN Exceptions e ON d.id = e.document_id
        WHERE d.status = 'exception';
    """

    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        exceptions = [dict(zip(columns, row)) for row in rows]

        return jsonify({"exceptions": exceptions}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

    """
    Response Structure
    {
        "exceptions": [
            {
            "documentId": 103,
            "fileName": "claim_103.pdf",
            "type": "Claim",
            "exceptionType": "Missing Data",
            "batchId": "Batch_0420A",
            "dateIdentified": "2025-04-19",
            "json": "{\"claimnumber\":\"\",\"name\":\"John Doe\"}"
            },
            {
            "documentId": 97,
            "fileName": "mortgage_97.pdf",
            "type": "Mortgage",
            "exceptionType": "Validation Error",
            "batchId": "Batch_0419B",
            "dateIdentified": "2025-04-18",
            "json": "{\"policynumber\":\"12345\"}"
            }
        ]
    }
    """

    


@app.route('/api/processed-documents', methods=['GET'])
def get_processed_documents():
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT 
            d.id,
            d.doc_name AS fileName,
            dt.type_name AS type,
            d.created_at AS processedDate,
            b.batch_name AS batchId,
            CASE
                WHEN d.status = 'processed' THEN 'Processed'
                WHEN d.status = 'verified' THEN 'Verified'
                ELSE d.status
            END AS status
        FROM Documents d
        JOIN DocumentTypes dt ON d.type_id = dt.id
        JOIN Batches b ON d.batch_id = b.id
        WHERE d.status IN ('processed', 'verified')
        ORDER BY d.created_at DESC
    """

    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        processed_documents = [dict(zip(columns, row)) for row in rows]

        return jsonify({"processed_documents": processed_documents}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()
    
    """
    {
        "processed_documents": [
            {
                "id": 105,
                "fileName": "claim_form_105.pdf",
                "type": "Claim",
                "processedDate": "2025-04-20T09:23:00",
                "batchId": "Batch_0420A",
                "status": "Processed"
            },
            {
                "id": 104,
                "fileName": "mortgage_doc_104.pdf",
                "type": "Mortgage",
                "processedDate": "2025-04-19T14:50:00",
                "batchId": "Batch_0419B",
                "status": "Verified"
            }
        ]
        }
    """

@app.route('/api/batches', methods=['GET'])
def get_batches():
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """ 
        SELECT 
            b.id AS batch_id,
            b.batch_name,
            DATE(b.created_at) AS date,
            (SELECT COUNT(*) FROM Documents WHERE Documents.batch_id = b.id) AS total_documents,
            CASE 
                WHEN EXISTS (
                    SELECT 1 FROM Documents 
                    WHERE Documents.batch_id = b.id AND Documents.status = 'exception'
                ) 
                THEN (
                    SELECT COUNT(*) FROM Documents 
                    WHERE Documents.batch_id = b.id AND Documents.status = 'exception'
                )
                ELSE 0
            END AS exceptions_count
        FROM Batches b;
    """
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        batches = [dict(zip(columns, row)) for row in rows]

        return jsonify({"batches": batches}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/api/exceptions_details', methods=['POST'])
def get_exception_details():
    try:
        conn = get_db_connection()
        # Extract the document_id from the request body
        data = request.get_json()
        print(data)
        document_id = data.get('document_id', "")
        
        if not document_id:
            return jsonify({'error': 'Document ID is required'}), 400

        # Get the document and its type
        cursor.execute("""
            SELECT d.id AS document_id, d.doc_name, dt.type_name, e.id AS exception_id, 
                   e.exception_message, e.resolved, e.created_at AS exception_created_at
            FROM Documents d
            LEFT JOIN DocumentTypes dt ON d.type_id = dt.id
            LEFT JOIN Exceptions e ON e.document_id = d.id
            WHERE d.id = %s
        """, (document_id,))
        doc_info = cursor.fetchone()

        if not doc_info:
            return jsonify({'error': 'Document not found'}), 404

        # Fetch type-specific details
        doc_type = doc_info[2].lower()  # Getting document type (index 2)
        type_specific_info = None

        if doc_type == 'mortgage':
            cursor.execute("SELECT * FROM MortgageDocuments WHERE document_id = %s", (document_id,))
        elif doc_type == 'claim':
            cursor.execute("SELECT * FROM ClaimDocuments WHERE document_id = %s", (document_id,))
        elif doc_type == 'coupon':
            cursor.execute("SELECT * FROM CouponDocuments WHERE document_id = %s", (document_id,))
        elif doc_type == 'check':
            cursor.execute("SELECT * FROM CheckDocuments WHERE document_id = %s", (document_id,))
        elif doc_type == 'agency':
            cursor.execute("SELECT * FROM AgencyDocuments WHERE document_id = %s", (document_id,))

        row = cursor.fetchone()
        type_specific_info = dict(zip([col[0] for col in cursor.description], row)) if row else None
        type_specific_info.pop('document_id', None)
        response = {
            'document': {
                'id': doc_info[0],
                'name': doc_info[1],
                'type': doc_type
            },
            'exception': {
                'id': doc_info[3],
                'message': doc_info[4],
                'created_at': doc_info[6]
            } if doc_info[3] else None,
            'type_specific_data': type_specific_info
        }
        conn.close()
        return jsonify(response)

    except mysql.connector.Error as e:  # Using the specific MySQL error class
        return jsonify({'error': str(e)}), 500
    except Exception as e:  # Catch other exceptions
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_json', methods=['POST'])
def update_json():
    try:
        data = request.json
        document_id = data.get('document_id')
        json_content = data.get('json_content')
        
        if not document_id or json_content is None:
            return jsonify({"status_code": 400, "message": "Missing document_id or json_content"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert JSON object to string for storage
        json_string = json.dumps(json_content)
        print('1')
        # Update the document's JSON in the database
        query = """
        UPDATE documents 
        SET documents.json = %s
        WHERE id = %s
        """
        cursor.execute(query, (json_string, document_id))
        conn.commit()

        cursor.close()
        conn.close()
        
        return jsonify({
            "status_code": 200,
            "message": "JSON updated successfully",
            "document_id": document_id
        })
    
    except Exception as e:
        print(f"Error updating JSON: {e}")
        return jsonify({
            "status_code": 500,
            "message": f"Error updating JSON: {str(e)}"
        }), 500




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=os.getenv('DEBUG'))