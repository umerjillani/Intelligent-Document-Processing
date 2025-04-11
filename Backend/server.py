from flask import Flask, request, jsonify
# from new import process_directory
from flask_cors import CORS
from document_processor import process_files
import os
import mysql.connector
from dbOps import get_db_connection
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

conn = mysql.connector.connect(
    host="localhost",      # Change this if your MySQL server is remote
    user="root",  # Replace with your MySQL username
    password="root",  # Replace with your MySQL password
    database="DocumentProcessing"  # Replace with your database name
)

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
    conn.close()
    print(rows)
    # Print results
    return rows

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    conn = get_db_connection()
    # SQL Queries
    queries = {
        "documents_today": """
            SELECT COUNT(*) AS documents_today 
            FROM Documents 
            WHERE DATE(created_at) = CURDATE();
        """,
        
        "processing_rate": """
            SELECT (COUNT(CASE WHEN status = 'processed' THEN 1 END) / COUNT(*)) * 100 
            AS processing_rate 
            FROM Documents;
        """,
        
        "pending_exceptions": """
            SELECT COUNT(*) AS pending_exceptions 
            FROM Exceptions 
            WHERE resolved = FALSE;
        """,
        
        "document_type_breakdown": """
            SELECT dt.type_name, COUNT(d.id) as count,
                   (COUNT(d.id) / (SELECT COUNT(*) FROM Documents)) * 100 as percentage
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
                    WHEN EXISTS (SELECT 1 FROM Documents WHERE batch_id = b.id AND status = 'exception') THEN 'Exceptions'
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
            result = cursor.fetchall()
            response_data[key] = result
        print(response_data)
        conn.close()
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Response Structure
    # {
    #     "documents_today": [{"documents_today": 124}],
    #     "processing_rate": [{"processing_rate": 85.0}],
    #     "pending_exceptions": [{"pending_exceptions": 18}],
    #     "active_batches": [{"active_batches": 5}],
    #     "recent_batches": [
    #         {"batch_id": "B-2025-0142", "documents": 32, "status": "Complete"},
    #         {"batch_id": "B-2025-0141", "documents": 18, "status": "Exceptions"},
    #         {"batch_id": "B-2025-0140", "documents": 45, "status": "Processing"},
    #         {"batch_id": "B-2025-0139", "documents": 29, "status": "Complete"}
    #     ]
    # }
    # print(response_data)
    
    # return jsonify(response_data)

@app.route('/api/exceptions', methods=['GET'])
def get_exceptions():  

    conn = get_db_connection()
    query = """SELECT
                d.id as documentId,
                d.doc_name as fileName,
                dt.type_name as type,
                e.exception_type as exceptionType,
                b.batch_name as batchId,
                DATE_FORMAT(e.created_at, '%Y-%m-%d') as dateIdentified,
                d.json as json
                FROM Documents d
                JOIN DocumentTypes dt ON d.type_id = dt.id
                JOIN Batches b ON d.batch_id = b.id
                JOIN Exceptions e ON d.id = e.document_id
                WHERE d.status = 'exception';"""
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    
    json = jsonify({"exceptions": data})
    conn.close()
    return json

@app.route('/api/processed-documents', methods=['GET'])
def get_processed_documents():
    conn = get_db_connection()
    query = """
        SELECT 
            d.id,
            CONCAT('DOC-', YEAR(d.created_at), '-', LPAD(d.id, 4, '0')) AS documentId,
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
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return jsonify({"processed_documents": data})


@app.route('/api/batches', methods=['GET'])
def get_batches():
    conn = get_db_connection()
    query = """ SELECT 
            b.id AS batch_id,
            b.batch_name,
            DATE(b.created_at) AS date,
            (SELECT COUNT(*) FROM Documents WHERE Documents.batch_id = b.id) AS total_documents,
            CASE 
                WHEN EXISTS (
                    SELECT 1 FROM Documents 
                    WHERE Documents.batch_id = b.id AND Documents.status = 'Exception'
                ) 
                THEN (
                    SELECT COUNT(*) FROM Documents 
                    WHERE Documents.batch_id = b.id AND Documents.status = 'Exception'
                )
                ELSE 0
            END AS exceptions_count
        FROM Batches b; """
    
    cursor.execute(query)
    data = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return jsonify({"batches": data})



# @app.route('/api/process_docs_path', methods=['POST'])
# def process_docs():
#     print("Aya")
    
#     # Check if the request contains JSON data
#     if not request.is_json:
#         return jsonify({"error": "Request must be JSON"}), 400

#     # Retrieve the list of file paths from the request
#     data = request.get_json()
#     file_paths = data.get('files', [])

#     # Check if 'file_paths' is provided and it's a non-empty list
#     if not file_paths or not isinstance(file_paths, list):
#         return jsonify({"error": "Invalid or missing 'file_paths'"}), 400

#     print(f"Processing {len(file_paths)} files...")

#     results = process_directory(file_paths, api_key)

#     if results:
#         return jsonify({"results": results}), 200
#     else:
#         return jsonify({"error": "No valid files processed"}), 400


@app.route('/api/exceptions/<string:document_id>/details', methods=['GET'])
def get_document_details(document_id):
    print('Hi')
    try:
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
        doc_type = doc_info['type_name']
        type_specific_info = None

        if doc_type == 'Mortgage':
            cursor.execute("SELECT * FROM MortgageDocuments WHERE document_id = %s", (document_id,))
            type_specific_info = cursor.fetchone()
        elif doc_type == 'Claim':
            cursor.execute("SELECT * FROM ClaimDocuments WHERE document_id = %s", (document_id,))
            type_specific_info = cursor.fetchone()
        elif doc_type == 'Coupon':
            cursor.execute("SELECT * FROM CouponDocuments WHERE document_id = %s", (document_id,))
            type_specific_info = cursor.fetchone()
        elif doc_type == 'Check':
            cursor.execute("SELECT * FROM CheckDocuments WHERE document_id = %s", (document_id,))
            type_specific_info = cursor.fetchone()
        elif doc_type == 'Agency':
            cursor.execute("SELECT * FROM AgencyDocuments WHERE document_id = %s", (document_id,))
            type_specific_info = cursor.fetchone()

        response = {
            'document': {
                'id': doc_info['document_id'],
                'name': doc_info['doc_name'],
                'type': doc_type
            },
            'exception': {
                'id': doc_info['exception_id'],
                'message': doc_info['exception_message'],
                'resolved': doc_info['resolved'],
                'created_at': doc_info['exception_created_at']
            } if doc_info['exception_id'] else None,
            'type_specific_data': type_specific_info
        }
        print(response)

        return jsonify(response)

    except Error as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exceptions_details', methods=['POST'])
def get_exception_details():
    try:
        conn = get_db_connection()
        # Extract the document_id from the request body
        data = request.get_json()
        print(data)
        document_id = data.get('document_id', "")
        document_id = document_id.split('-')[1]
        
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
        
        print('2')
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

@app.route('/api/update_document', methods=['PUT'])
def update_document_api():
    try:
        conn = get_db_connection()
        # Get JSON data from request
        data = request.get_json()
        # Extract required fields
        document_id = data.get('document_id').split('-')[1]
        document_type = data.get('document_type', '').lower()
        updated_fields = data.get('fields', {})
        print(document_id)
        print(document_type)
        print(updated_fields)
        # Validate required fields
        if not document_id or not document_type:
            return jsonify({'error': 'Document ID and type are required'}), 400

        # Check if document exists and matches the claimed type
        cursor.execute("""
            SELECT d.id 
            FROM Documents d
            JOIN DocumentTypes dt ON d.type_id = dt.id
            WHERE d.id = %s AND LOWER(dt.type_name) = %s
        """, (document_id, document_type))
        

        if not cursor.fetchone():
            return jsonify({'error': 'Document not found or type mismatch'}), 404
        # Call the generic update function with the provided parameters
        return update_document_generic(document_id, document_type, updated_fields)
            
    except mysql.connector.Error as e:
        # Handle database errors
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        # Handle other errors
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    finally:
        conn.close()

# Document type field configurations
DOCUMENT_FIELD_CONFIG = {
    'mortgage': {
        'fields': {
            'policynumber': {'type': 'string', 'required': True},
            'name': {'type': 'string', 'required': True},
            'address': {'type': 'string', 'required': True}
        },
        'table': 'MortgageDocuments'
    },
    'claim': {
        'fields': {
            'claimnumber': {'type': 'string', 'required': True},
            'name': {'type': 'string', 'required': True},
            'address': {'type': 'string', 'required': True}
        },
        'table': 'ClaimDocuments'
    },
    'coupon': {
        'fields': {
            'policynumber': {'type': 'string', 'required': True},
            'insured': {'type': 'string', 'required': True},
            'amount': {'type': 'decimal', 'required': True}
        },
        'table': 'CouponDocuments'
    },
    'check': {
        'fields': {
            'policynumber': {'type': 'string', 'required': True},
            'loannumber': {'type': 'string', 'required': True},
            'amount': {'type': 'decimal', 'required': True}
        },
        'table': 'CheckDocuments'
    },
    'agency': {
        'fields': {
            'agencyname': {'type': 'string', 'required': True},
            'producer': {'type': 'string', 'required': True}
        },
        'table': 'AgencyDocuments'
    }
}

# Generic document update function using the config
def update_document_generic(document_id, document_type, updated_fields):
    try:
        conn.get_db_connection()
        print('Enter')
        if document_type not in DOCUMENT_FIELD_CONFIG:
            return jsonify({'error': 'Invalid document type'}), 400
            
        config = DOCUMENT_FIELD_CONFIG[document_type]
        table_name = config['table']
        valid_fields = config['fields']
        
        # Validate and filter fields
        set_clause_parts = []
        params = []
        print('Updation Start')
        
        for field, value in updated_fields.items():
            if field in valid_fields:
                # Additional type validation could be added here
                set_clause_parts.append(f"{field} = %s")
                params.append(value)
        print('Updation End')
        
        if not set_clause_parts:
            return jsonify({'error': 'No valid fields to update'}), 400
            
        # Construct and execute query
        set_clause = ", ".join(set_clause_parts)
        print(set_clause)
        print(table_name)
        query = f"UPDATE {table_name} SET {set_clause} WHERE document_id = %s"
        print(query)
        params.append(document_id)
        print(tuple(params))
        cursor.execute(query, tuple(params))
        print('Update Statement')
        conn.commit()
        
        return jsonify({'success': True, 'message': f'{document_type.capitalize()} document updated successfully'})
        
    except mysql.connector.Error as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)