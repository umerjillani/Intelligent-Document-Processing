import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
import os
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

def ensure_document_types_in_db(categories):
    """
    Ensure all document categories exist in the DocumentTypes table
    
    Args:
        categories: List of document categories to check/create
    
    Returns:
        dict: Mapping of category names to their IDs in the database
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get existing document types
        cursor.execute("SELECT id, type_name FROM DocumentTypes")
        existing_types = {row[1].lower(): row[0] for row in cursor.fetchall()}
        
        # Add any missing types
        for category in categories:
            if category.lower() not in existing_types:
                cursor.execute(
                    "INSERT INTO DocumentTypes (type_name) VALUES (%s)",
                    (category,)
                )
                conn.commit()
                
                # Get the ID of the newly inserted type
                cursor.execute(
                    "SELECT id FROM DocumentTypes WHERE type_name = %s",
                    (category,)
                )
                type_id = cursor.fetchone()[0]
                existing_types[category.lower()] = type_id
        
        cursor.close()
        conn.close()
        
        return existing_types
    except Exception as e:
        logger.error(f"Database error ensuring document types: {str(e)}")
        raise

def create_batch_in_db(batch_name):
    """
    Create a new batch record in the database
    
    Args:
        batch_name: Name for the new batch
    
    Returns:
        int: The ID of the newly created batch
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create a new batch entry
        cursor.execute(
            "INSERT INTO Batches (batch_name) VALUES (%s)",
            (batch_name,)
        )
        
        # Get the ID of the newly inserted batch
        conn.commit()
        cursor.execute(
            "SELECT id FROM Batches WHERE batch_name = %s",
            (batch_name,)
        )
        batch_id = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return batch_id
    except Exception as e:
        logger.error(f"Database error creating batch: {str(e)}")
        raise

# def insert_document_in_db(batch_id, doc_name, type_id, status):
#     """
#     Insert a document record into the database
    
#     Args:
#         batch_id: ID of the batch this document belongs to
#         doc_name: Name of the document file
#         type_id: ID of the document type from DocumentTypes table
#         status: Processing status ('pending', 'processed', or 'exception')
    
#     Returns:
#         int: The ID of the newly created document record
#     """
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
        
#         cursor.execute(
#             """
#             INSERT INTO Documents (
#                 batch_id,
#                 doc_name,
#                 type_id,
#                 status
#             ) VALUES (%s, %s, %s, %s)
#             """,
#             (batch_id, doc_name, type_id, status)
#         )
        
#         # Get the ID of the newly inserted document
#         conn.commit()
#         cursor.execute(
#             "SELECT LAST_INSERT_ID()"
#         )
#         doc_id = cursor.fetchone()[0]
        
#         cursor.close()
#         conn.close()
        
#         return doc_id
#     except Exception as e:
#         logger.error(f"Database error inserting document: {str(e)}")
#         raise

def insert_document_in_db(batch_id, doc_name, type_id, status, file_url=None, json_content=None):
    """
    Insert a document record into the database

    Args:
        batch_id: ID of the batch this document belongs to
        doc_name: Name of the document file
        type_id: ID of the document type from DocumentTypes table
        status: Processing status ('pending', 'processed', or 'exception')
        file_url: S3 URL or path of the file
        json_content: JSON content as string (optional)

    Returns:
        int: The ID of the newly created document record
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if json_content:
            cursor.execute(
                """
                INSERT INTO Documents (
                    batch_id, doc_name, type_id, status, file, json
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (batch_id, doc_name, type_id, status, file_url, json_content)
            )
        else:
            cursor.execute(
                """
                INSERT INTO Documents (
                    batch_id, doc_name, type_id, status, file
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (batch_id, doc_name, type_id, status, file_url)
            )

        conn.commit()
        cursor.execute("SELECT LAST_INSERT_ID()")
        doc_id = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return doc_id
    except Exception as e:
        logger.error(f"Database error inserting document: {str(e)}")
        raise


def add_exception_in_db(document_id, exception_message):
    """
    Insert an exception record into the database
    
    Args:
        document_id: ID of the document with the exception
        exception_message: Message describing the exception
    
    Returns:
        int: The ID of the newly created exception record
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO Exceptions (
                document_id,
                exception_message
            ) VALUES (%s, %s)
            """,
            (document_id, exception_message)
        )
        
        # Get the ID of the newly inserted exception
        conn.commit()
        cursor.execute(
            "SELECT LAST_INSERT_ID()"
        )
        exception_id = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return exception_id
    except Exception as e:
        logger.error(f"Database error adding exception: {str(e)}")
        raise

def get_db_connection():
    """
    Create and return a database connection
    
    Returns:
        connection: Database connection object
    """
    # Replace with your actual database connection code
    import mysql.connector
    
    return mysql.connector.connect(**db_config)


def insert_check_document(document_id, policynumber, loannumber, amount):
    """
    Insert check document details into the CheckDocuments table
    
    Args:
        document_id: ID of the document in the Documents table
        policynumber: Policy number from the document
        loannumber: Loan number from the document
        amount: Monetary amount from the document
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO CheckDocuments (document_id, policynumber, loannumber, amount)
        VALUES (%s, %s, %s, %s)
        """
        
        cursor.execute(query, (document_id, policynumber, loannumber, amount))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully inserted check document details for document ID {document_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to insert check document details: {str(e)}")
        raise

def insert_mortgage_document(document_id, policynumber, name, address):
    """
    Insert mortgage document details into the MortgageDocuments table
    
    Args:
        document_id: ID of the document in the Documents table
        policynumber: Policy number from the document
        name: Name from the document
        address: Address from the document
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO MortgageDocuments (document_id, policynumber, name, address)
        VALUES (%s, %s, %s, %s)
        """
        
        cursor.execute(query, (document_id, policynumber, name, address))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully inserted mortgage document details for document ID {document_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to insert mortgage document details: {str(e)}")
        raise

def insert_claim_document(document_id, claimnumber, name, address):
    """
    Insert claim document details into the ClaimDocuments table
    
    Args:
        document_id: ID of the document in the Documents table
        claimnumber: Claim number from the document
        name: Name from the document
        address: Address from the document
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO ClaimDocuments (document_id, claimnumber, name, address)
        VALUES (%s, %s, %s, %s)
        """
        
        cursor.execute(query, (document_id, claimnumber, name, address))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully inserted claim document details for document ID {document_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to insert claim document details: {str(e)}")
        raise

def insert_coupon_document(document_id, policynumber, insured, amount):
    """
    Insert coupon document details into the CouponDocuments table
    
    Args:
        document_id: ID of the document in the Documents table
        policynumber: Policy number from the document
        insured: Insured name from the document
        amount: Monetary amount from the document
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO CouponDocuments (document_id, policynumber, insured, amount)
        VALUES (%s, %s, %s, %s)
        """
        
        cursor.execute(query, (document_id, policynumber, insured, amount))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully inserted coupon document details for document ID {document_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to insert coupon document details: {str(e)}")
        raise

def insert_agency_document(document_id, agencyname, producer):
    """
    Insert agency document details into the AgencyDocuments table
    
    Args:
        document_id: ID of the document in the Documents table
        agencyname: Agency name from the document
        producer: Producer name from the document
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO AgencyDocuments (document_id, agencyname, producer)
        VALUES (%s, %s, %s)
        """
        
        cursor.execute(query, (document_id, agencyname, producer))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully inserted agency document details for document ID {document_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to insert agency document details: {str(e)}")
        raise