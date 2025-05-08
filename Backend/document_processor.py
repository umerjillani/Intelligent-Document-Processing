import os
import re
import json
import shutil
import logging
from decimal import Decimal, InvalidOperation, getcontext, ROUND_HALF_UP
import importlib.util
import tempfile
from PIL import Image
import pytesseract
import fitz
import io
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import openai
import numpy as np
import cv2
import torch
from datetime import datetime
import ast
from dotenv import load_dotenv
import boto3
import botocore.exceptions

# Load environment variables from .env file
load_dotenv()
print('1')
# Set up S3 client
s3_client = boto3.client('s3')
bucket_name = "nfipdirectories"

print('2')
# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

# Check required imports
REQUIRED_PACKAGES = [
    "PIL", "pytesseract", "fitz", "transformers", "openai", "cv2", "numpy", "torch"
]

missing_packages = []
for package in REQUIRED_PACKAGES:
    if importlib.util.find_spec(package) is None:
        missing_packages.append(package)

if missing_packages:
    error_msg = f"Missing required packages: {', '.join(missing_packages)}. Please install them using pip."
    logger.error(error_msg)
    raise ImportError(error_msg)

# Set precision for Decimal calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

# Verify S3 connectivity
def verify_s3_connectivity():
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
        return True
    except botocore.exceptions.ClientError as e:
        logger.error(f"Failed to connect to S3 bucket {bucket_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error verifying S3 connectivity: {str(e)}")
        return False

def create_s3_directory(path):
    try:
        if not path.endswith("/"):
            path += "/"
        s3_client.put_object(Bucket=bucket_name, Key=path)
        logger.info(f"Created S3 directory: {path}")
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        raise

def verify_s3_directory(path):
    """Verify if an S3 directory exists, create it if it doesn't."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=path.rstrip("/") + "/")
        logger.debug(f"S3 directory exists: {path}")
    except s3_client.exceptions.ClientError:
        logger.info(f"S3 directory {path} does not exist, creating it")
        create_s3_directory(path)
    except Exception as e:
        logger.error(f"Error verifying directory {path}: {e}")
        raise

def list_s3_files(prefix):
    try:
        logger.info(f"Listing objects in S3 bucket: {bucket_name}, prefix: {prefix}")
        files = []
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        all_objects = []
        for page in page_iterator:
            contents = page.get("Contents", [])
            all_objects.extend([item["Key"] for item in contents])

        logger.info(f"All objects in {prefix}: {all_objects}")

        for key in all_objects:
            if not key.endswith("/") and key.lower().endswith((".tif", ".tiff", ".pdf")):
                files.append(key)

        if not files:
            logger.warning(f"No valid TIFF or PDF files found in bucket: {bucket_name}, prefix: {prefix}")
        else:
            logger.info(f"Found {len(files)} valid TIFF/PDF files in {prefix}: {files}")
        return files
    except botocore.exceptions.ClientError as e:
        logger.error(f"Client error listing files in {prefix}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error listing files in {prefix}: {str(e)}")
        raise

def download_file_from_s3(s3_path, local_path):
    try:
        s3_client.download_file(bucket_name, s3_path, local_path)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Downloaded file {local_path} not found")
        logger.info(f"Downloaded {s3_path} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {s3_path} to {local_path}: {e}")
        raise

def upload_file_to_s3(local_path, s3_path):
    try:
        if not os.path.isfile(local_path):
            logger.error(f"Local file {local_path} does not exist for upload to {s3_path}")
            raise FileNotFoundError(f"Local file {local_path} does not exist")
        s3_path = s3_path.replace("\\", "/")
        s3_client.upload_file(local_path, bucket_name, s3_path)
        logger.info(f"Uploaded {local_path} to S3 as {s3_path}")
        url = f"https://{bucket_name}.s3.amazonaws.com/{s3_path}"
        return url
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to {s3_path}: {e}")
        raise

def upload_fileobj_to_s3(file_obj, s3_path):
    try:
        file_obj.seek(0)  # Reset file pointer to beginning
        s3_client.upload_fileobj(file_obj, bucket_name, s3_path)
        logger.info(f"Uploaded file object to S3 as {s3_path}")
        url = f"https://{bucket_name}.s3.amazonaws.com/{s3_path}"
        return url
    except Exception as e:
        logger.error(f"Failed to upload file object to {s3_path}: {e}")
        raise

print('3')
try:
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large", use_fast=False)
    print('3a')
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")
    print('3b')
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    print('3c')
except Exception as e:
    logger.error(f"Failed to load NLP models: {str(e)}")
    raise

CATEGORY_LABELS = ["Claim", "Mortgage", "No Money - Cancel", "Check", "Coupons", "Agency"]

category_variations = {
    "check": {
        "policyNumber": {"variations": {"policy", "policyNumber", "policyNo", "memo", "memo#"}, "essential": True},
        "loanNumber": {"variations": {"loan", "loanNumber", "loanNo"}, "essential": True},
        "amount": {"variations": {"amount", "amt", "checkAmount", "chequeAmount", "totalAmount", "netAmount", "netAmt"}, "essential": True}
    },
    "mortgage": {
        "policyNumber": {"variations": {"policy", "policyNumber", "policyNo"}, "essential": True},
        "name": {"variations": {"name", "clientName", "client"}, "essential": True},
        "address": {"variations": {"address", "propertyAddress", "property"}, "essential": True}
    },
    "claim": {
        "claimNumber": {"variations": {"claimNumber", "claim", "claimNo", "claimNo."}, "essential": True},
        "name": {"variations": {"name", "clientName", "client"}, "essential": True},
        "address": {"variations": {"address", "propertyAddress", "property"}, "essential": True}
    },
    "coupons": {
        "policyNumber": {"variations": {"policyNumber", "policy", "policyNumberId", "policyNo"}, "essential": True},
        "insured": {"variations": {"insured", "insuredName"}, "essential": True},
        "amount": {"variations": {"amount", "amt", "checkAmount", "chequeAmount", "totalAmount", "netAmount", "netAmt"}, "essential": True}
    },
    "agency": {
        "agencyName": {"variations": {"agency", "agencyName", "insuranceAgency"}, "essential": True},
        "producer": {"variations": {"producer", "producerName", "agent"}, "essential": True}
    }
}

REFERENCE_FILE = "Files/no_money_reference.png"  # Explicitly a PNG

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

def extract_float_value(value):
    if isinstance(value, (int, float)):
        return Decimal(str(float(value)))

    cleaned_value = str(value).replace('$', '').replace(',', '').replace('USD', '').strip()
    if '.' not in cleaned_value and cleaned_value.count(',') == 1:
        cleaned_value = cleaned_value.replace(',', '.')

    match = re.search(r'-?\d*\.?\d+', cleaned_value)
    if not match:
        return Decimal('0.0')

    try:
        return Decimal(match.group())
    except (InvalidOperation, TypeError):
        return Decimal('0.0')

def clean_ocr_text(text):
    if not text:
        return ""
    try:
        text = text.encode('utf-8', 'ignore').decode('utf-8')
    except UnicodeError:
        text = ''.join(char for char in text if ord(char) < 128)
    text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_tiff(tiff_path):
    try:
        logger.info(f"Performing OCR on TIFF: {tiff_path}")
        img = Image.open(tiff_path)
        text = ""
        for i in range(img.n_frames):
            try:
                img.seek(i)
                original_dpi = img.info.get("dpi", (72, 72))
                if isinstance(original_dpi, tuple) and len(original_dpi) == 2:
                    x_dpi, y_dpi = original_dpi
                else:
                    x_dpi = y_dpi = 72
                frame = img.convert("RGB") if img.mode != "RGB" else img
                x_scale = 300 / x_dpi
                y_scale = 300 / y_dpi
                if x_scale != 1 or y_scale != 1:
                    new_width = int(frame.width * x_scale)
                    new_height = int(frame.height * y_scale)
                    frame = frame.resize((new_width, new_height), Image.LANCZOS)
                page_text = pytesseract.image_to_string(frame)
                text += page_text + "\n"
            except Exception as e:
                logging.error(f"Error processing frame {i} of TIFF {tiff_path}: {str(e)}")
                continue
        return clean_ocr_text(text)
    except Exception as e:
        logging.error(f"TIFF extraction failed for {tiff_path}: {str(e)}")
        return ""

def extract_text_from_fileobj(file_obj, file_type):
    """Extract text from a file object based on its type (TIFF or PDF)"""
    try:
        if file_type.lower() in ('.tif', '.tiff'):
            with tempfile.NamedTemporaryFile(suffix=file_type, delete=False) as temp_file:
                file_obj.seek(0)
                shutil.copyfileobj(file_obj, temp_file)
                temp_path = temp_file.name
            
            text = extract_text_from_tiff(temp_path)
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            return text
        elif file_type.lower() == '.pdf':
            file_obj.seek(0)
            text = extract_text_from_pdf_fileobj(file_obj)
            return text
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Text extraction failed for file object: {str(e)}")
        return ""

print('4')
def extract_text_from_pdf_fileobj(file_obj):
    try:
        file_obj.seek(0)
        pdf_document = fitz.open(stream=file_obj.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text("text").strip()
            text += page_text + "\n" if page_text else ""
        text = clean_ocr_text(text)
        
        # If text extraction doesn't yield good results, try OCR
        if not text or len(text) < 50:
            text = ""
            file_obj.seek(0)
            pdf_document = fitz.open(stream=file_obj.read(), filetype="pdf")
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    page_text = pytesseract.image_to_string(img)
                    text += page_text + "\n"
                except Exception as e:
                    logging.error(f"Error performing OCR on page {page_num} of PDF: {str(e)}")
                    continue
            text = clean_ocr_text(text)
        pdf_document.close()
        return text
    except Exception as e:
        logging.error(f"PDF text extraction failed: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text("text").strip()
            text += page_text + "\n" if page_text else ""
        text = clean_ocr_text(text)
        
        # If text extraction doesn't yield good results, try OCR
        if not text or len(text) < 50:
            text = ""
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    page_text = pytesseract.image_to_string(img)
                    text += page_text + "\n"
                except Exception as e:
                    logging.error(f"Error performing OCR on page {page_num} of PDF {pdf_path}: {str(e)}")
                    continue
            text = clean_ocr_text(text)
        pdf_document.close()
        return text
    except Exception as e:
        logging.error(f"PDF text extraction failed for {pdf_path}: {str(e)}")
        return ""

def image_to_gray(img_path, page_num=0):
    try:
        if img_path.lower().endswith(".png"):  # Reference PNG (single-frame)
            img = Image.open(img_path).convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        elif img_path.lower().endswith((".tif", ".tiff")):  # Target TIFF
            img = Image.open(img_path).convert("RGB")
            if hasattr(img, 'n_frames') and img.n_frames > page_num:
                img.seek(page_num)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        elif img_path.lower().endswith(".pdf"):  # Target PDF
            pdf = fitz.open(img_path)
            pix = pdf[page_num].get_pixmap(dpi=300) if page_num < len(pdf) else pdf[0].get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
            pdf.close()
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported file format: {img_path}")
    except Exception as e:
        logger.error(f"Failed to convert {img_path} to grayscale (page {page_num}): {str(e)}")
        raise

def preprocess_image(img):
    """Preprocess image to enhance template matching robustness."""
    # Normalize to handle brightness/contrast variations
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # Apply adaptive thresholding to binarize and reduce noise
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Slight blur to smooth out noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def template_match(reference_img, target_img, threshold=0.7):
    """Perform template matching between reference and target images."""
    try:
        # Preprocess both images
        ref_processed = preprocess_image(reference_img)
        target_processed = preprocess_image(target_img)
        
        # Resize target to match reference dimensions if needed
        if target_processed.shape != ref_processed.shape:
            target_processed = cv2.resize(target_processed, (ref_processed.shape[1], ref_processed.shape[0]))
        
        # Perform template matching (using normalized correlation coefficient)
        result = cv2.matchTemplate(target_processed, ref_processed, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        return max_val  # Return the highest correlation score
    except Exception as e:
        logger.error(f"Template matching failed: {str(e)}")
        return 0.0

def is_similar_to_reference(file_path, match_threshold=0.7):
    """
    Compare target file to reference PNG using robust template matching.
    Args:
        file_path: Path to the target TIFF or PDF file.
        match_threshold: Minimum correlation score for a match.
    Returns:
        bool: True if the file is similar to the reference, False otherwise.
    """
    try:
        if not os.path.exists(REFERENCE_FILE):
            raise FileNotFoundError(f"Reference file not found: {REFERENCE_FILE}")
        if not REFERENCE_FILE.lower().endswith(".png"):
            raise ValueError("Reference file must be a PNG")
        if not file_path.lower().endswith((".tif", ".tiff", ".pdf")):
            raise ValueError("Target file must be TIFF or PDF")
        
        # Load reference image
        ref_img = image_to_gray(REFERENCE_FILE, 0)
        
        # Load target image (page 1)
        test_img1 = image_to_gray(file_path, 0)
        
        # Check for second page
        has_second_page = False
        if file_path.lower().endswith((".tif", ".tiff")):
            with Image.open(file_path) as img:
                has_second_page = hasattr(img, 'n_frames') and img.n_frames > 1
        elif file_path.lower().endswith(".pdf"):
            with fitz.open(file_path) as pdf:
                has_second_page = pdf.page_count > 1
        
        test_img2 = image_to_gray(file_path, 1) if has_second_page else test_img1
        
        # Perform template matching
        score1 = template_match(ref_img, test_img1, threshold=match_threshold)
        score2 = template_match(ref_img, test_img2, threshold=match_threshold) if has_second_page else score1
        
        # Check if both pages exceed the threshold
        is_similar = score1 >= match_threshold and score2 >= match_threshold
        
        logger.info(f"Similarity for {file_path}: "
                    f"Match1={score1:.2f}, Match2={score2:.2f}, Similar={is_similar}")
        return is_similar
    except Exception as e:
        logger.error(f"Image similarity check failed for {file_path}: {str(e)}")
        return False

def quick_classify(text):
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in ["claim number", "claim no", "claim id"]):
        return "Claim"
    elif any(phrase in text_lower for phrase in ["pay to the order of", "pay to the", "permat", "check number", "check no", "check #", "check id"]):
        return "Check"
    return None

def classify_document(text):
    logger.info("Starting document classification\n")
    quick_result = quick_classify(text)
    if quick_result:
        logger.info(f"Classified by Quick_Classify as {quick_result}")
        print(f"Classified by Quick_Classify as {quick_result}.\n")
        return quick_result
    
    text_to_classify = " ".join(text.split()[:2000])
    
    try:
        logger.info("Attempting Zero-shot classification")
        hypotheses = {
            "Check": "This document is a payment instrument with 'pay to the order of' text, dollar amounts, check numbers, bank routing numbers, and signatures. It includes payee name, account information, and monetary values.",
            "Coupons": "This document contains payment vouchers, premium notices, discount offers, or detachable bill payment stubs with amount due, payment instructions, and possibly tear-off sections. It likely includes policy numbers and payment deadlines.",
            "No Money - Cancel": "This document indicates a cancelled transaction, policy termination, void notification, insufficient funds notice, or cancellation confirmation. It likely mentions fees, cancellation dates, or refund information.",
            "Claim": "This document is an insurance claim form with a claim number, date of loss, insured party details, property damage descriptions, adjuster information, and claim status. It includes specific loss dates, claim IDs, and damage descriptions.",
            "Agency": "This document relates to insurance agency operations, containing agent licensing information, producer codes, commission statements, agency agreements, or company appointments. It includes agency names, producer information, and contract terms, but lacks specific claim numbers or loss details.",
            "Mortgage": "This document pertains to property financing, containing loan numbers, principal amounts, interest rates, property addresses, lender details, and payment schedules. It includes deed information, closing details, and property descriptions."
        }
        
        candidate_labels = list(hypotheses.keys())
        candidate_hypotheses = list(hypotheses.values())
        result = classifier(text_to_classify, candidate_labels=candidate_labels, hypothesis=candidate_hypotheses, multi_label=False)
        
        if result['scores'][0] > 0.70:
            logger.info(f"Classified by Zero-shot as {result['labels'][0]} with confidence = {result['scores'][0]}")
            print(f"Classified by Zero-shot as {result['labels'][0]} with confidence = {result['scores'][0]}\n")
            return result['labels'][0]
    except Exception as e:
        logger.warning(f"Zero-shot classification failed: {str(e)}")
    
    api_key = os.getenv("OPENAI_API")
    if not api_key:
        logger.error("OpenAI API key not found. Cannot perform classification.")
        return "Unknown"
    
    try:
        logger.info("Attempting OpenAI classification")
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "Deeply analyze the entire content (OCR text) and classify this document into one category:\n"
            "- Check: Payment checks with 'Pay to the order of' text, dollar amounts, check numbers, bank routing numbers, payee names, and signatures.\n"
            "- Coupons: Payment vouchers, premium notices, detachable bill payment stubs with amount due, payment instructions, policy numbers, and possibly tear-off sections.\n"
            "- No Money - Cancel: Cancelled transaction notices, policy terminations, void notifications, insufficient funds notices with cancellation dates, fee information, or refund details.\n"
            "- Claim: Insurance claim documents with distinct claim numbers, date of loss information, damage descriptions, claim status updates, adjuster details, and references to specific damage incidents or losses.\n"
            "- Agency: Insurance agency administrative documents containing agent licensing info, producer codes, agency agreements, commission statements, without specific claim numbers or loss details. References to agency operations rather than individual claims.\n"
            "- Mortgage: Property financing documents with loan numbers, principal amounts, interest rates, property addresses, lender details, deed information, and payment schedules.\n\n"
            f"Document: {text[:1000]}\n\nCategory:"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
        
        raw_category = response.choices[0].message.content.strip()
        valid_categories = {cat.lower(): cat for cat in CATEGORY_LABELS}
        normalized_response = raw_category.lower()
        
        if normalized_response in valid_categories:
            category = valid_categories[normalized_response]
            logger.info(f"Classified by OpenAI as {category}")
            print(f"Classified by OpenAI as {category}")
            return category
        
        for valid_key in valid_categories:
            if valid_key in normalized_response:
                category = valid_categories[valid_key]
                logger.info(f"Classified by OpenAI as {category} (from partial match: '{raw_category}')")
                print(f"Classified by OpenAI as {category} (from partial match: '{raw_category}')")
                return category
        
        logger.warning(f"OpenAI returned unrecognized category: '{raw_category}'")
        print(f"OpenAI returned unrecognized category: '{raw_category}', defaulting to 'Unknown'")
        return "Unknown"
    except Exception as e:
        logger.error(f"OpenAI classification failed: {str(e)}")
        return "Unknown"

def normalize_string(value):
    if not value: return ""
    if not isinstance(value, str): value = str(value)
    return re.sub(r'[^a-zA-Z0-9]', '', value).lower().strip()

def to_camel_case(s):
    if not s:
        return s
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s).strip()
    words = []
    current_word = ""
    for i, char in enumerate(s):
        if char.isupper() and current_word:
            words.append(current_word.lower())
            current_word = char
        else:
            current_word += char
    if current_word:
        words.append(current_word.lower())
    words = [word for w in words for word in w.split() if word]
    if not words:
        return s.lower()
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

def convert_keys_to_camel_case(data):
    if isinstance(data, dict):
        return {to_camel_case(k): convert_keys_to_camel_case(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_camel_case(item) for item in data]
    return data

def search_key(data, target_keys, key_variations):
    if isinstance(target_keys, str):
        target_keys = [target_keys]
    variation_map = {normalize_string(key): variations for key, variations in key_variations.items()}
    stack = [data]
    found_values = {}
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in reversed(list(current.items())):
                norm_key = normalize_string(key)
                for target, variations in variation_map.items():
                    if norm_key in variations and value:
                        if target in [normalize_string(k) for k in ["amount", "checkAmount", "totalAmount", "netAmount"]]:
                            try:
                                amount = extract_float_value(value)
                                found_values[target] = max(found_values.get(target, Decimal('-inf')), amount)
                            except (InvalidOperation, TypeError) as e:
                                logger.warning(f"Could not convert amount value '{value}': {str(e)}")
                                continue
                        elif target not in found_values:
                            found_values[target] = str(value).strip()
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(reversed(current))
    return found_values

def safe_calculate_percentage(numerator, denominator):
    try:
        if not denominator or Decimal(denominator) == Decimal('0'):
            return Decimal('0')
        return (Decimal(numerator) / Decimal(denominator)) * Decimal('100')
    except (InvalidOperation, TypeError):
        return Decimal('0')

def safe_copy_file_to_s3(src_path, dest_s3_path):
    """Safely copy a file from local path to S3"""
    try:
        if not os.path.isfile(src_path):
            logger.error(f"Source file {src_path} does not exist for upload to {dest_s3_path}")
            return False
        url = upload_file_to_s3(src_path, dest_s3_path)
        logger.info(f"Copied {src_path} to S3 at {dest_s3_path}")
        return url
    except Exception as e:
        logger.error(f"Failed to copy file from {src_path} to {dest_s3_path}: {str(e)}")
        return False

def safe_copy_fileobj_to_s3(file_obj, dest_s3_path):
    """Safely copy a file object to S3"""
    try:
        url = upload_fileobj_to_s3(file_obj, dest_s3_path)
        logger.info(f"Copied file object to S3 at {dest_s3_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy file object to {dest_s3_path}: {str(e)}")
        return False

def save_fileobj_to_temp_and_process(file_obj, filename):
    """Save file object to a temporary file and return the path"""
    file_ext = os.path.splitext(filename)[1].lower()
    try:
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            file_obj.seek(0)
            shutil.copyfileobj(file_obj, temp_file)
            temp_path = temp_file.name
        return temp_path
    except Exception as e:
        logger.error(f"Failed to save file object to temporary file: {str(e)}")
        return None

def from_dbOps_import():
    """Import database operations functions"""
    try:
        from dbOps import (
            ensure_document_types_in_db, 
            create_batch_in_db, 
            insert_document_in_db, 
            add_exception_in_db, 
            get_db_connection, 
            insert_agency_document, 
            insert_coupon_document, 
            insert_claim_document, 
            insert_mortgage_document, 
            insert_check_document
        )
        return {
            "ensure_document_types_in_db": ensure_document_types_in_db,
            "create_batch_in_db": create_batch_in_db,
            "insert_document_in_db": insert_document_in_db,
            "add_exception_in_db": add_exception_in_db,
            "get_db_connection": get_db_connection,
            "insert_agency_document": insert_agency_document,
            "insert_coupon_document": insert_coupon_document,
            "insert_claim_document": insert_claim_document,
            "insert_mortgage_document": insert_mortgage_document,
            "insert_check_document": insert_check_document
        }
    except ImportError as e:
        logger.error(f"Failed to import database operations: {str(e)}")
        return None

def process_files(input_files):
    """
    Process files, categorize them, extract information, and store results.
    
    Args:
        input_files: List of file objects (with filename and file attributes)
        
    Returns:
        dict: Result containing status code, message, and statistics
    """
    # Ensure S3 connectivity before proceeding
    if not verify_s3_connectivity():
        logger.critical("Cannot proceed without S3 connectivity")
        return {
            "status_code": 500,
            "message": "S3 connectivity failed",
            "total_check_amount": 0,
            "successful_docs": 0,
            "exception_docs": 0
        }
    
    api_key = os.getenv('OPENAI_API')
    if not api_key:
        logger.error("API key not found. Cannot proceed with document processing.")
        return {
            "status_code": 400,
            "message": "API key not found. Cannot proceed with document processing.",
            "total_check_amount": 0,
            "successful_docs": 0,
            "exception_docs": 0
        }
    
    # Import database operations
    db_ops = from_dbOps_import()
    if not db_ops:
        logger.error("Failed to import database operations. Cannot proceed.")
        return {
            "status_code": 500,
            "message": "Failed to import database operations",
            "total_check_amount": 0,
            "successful_docs": 0,
            "exception_docs": 0
        }
    
    # Initialize counters and tracking variables
    successful_docs = 0
    exception_docs = 0
    check_amounts = []
    exceptions_list = []
    
    base_dir = "Files"
    categories = ["check", "mortgage", "claim", "coupons", "agency", "no money - cancel", "unknown"]
    output_dir = os.path.join(base_dir, "JSONs")
    
    # Create S3 directory structure
    try:
        logger.info("Verifying S3 directory structure")
        verify_s3_directory(os.path.join(base_dir, "RAW Data"))
        verify_s3_directory(output_dir)
        
        for cat in categories:
            if cat == "unknown":
                verify_s3_directory(os.path.join(base_dir, cat))
            else:
                cat_dir = cat.capitalize() if cat != "no money - cancel" else cat
                for status in ["Passed", "Failed"]:
                    verify_s3_directory(os.path.join(base_dir, cat_dir, status, "Files"))
                    verify_s3_directory(os.path.join(base_dir, cat_dir, status, "JSON"))
    except Exception as e:
        logger.error(f"Failed to create S3 directory structure: {str(e)}")
        return {
            "status_code": 500,
            "message": f"Failed to create S3 directory structure: {str(e)}",
            "total_check_amount": 0,
            "successful_docs": 0,
            "exception_docs": 0
        }
    
    # Ensure document type IDs are in the database
    try:
        category_ids_map = db_ops["ensure_document_types_in_db"](categories)
    except Exception as e:
        logger.error(f"Failed to ensure document types in database: {str(e)}")
        return {
            "status_code": 500,
            "message": f"Failed to prepare document types: {str(e)}",
            "total_check_amount": 0,
            "successful_docs": 0,
            "exception_docs": 0
        }
    
    json_tif_pairs = []
    
    # Generate a unique batch name based on timestamp
    batch_name = f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create a batch in the database
    try:
        batch_id = db_ops["create_batch_in_db"](batch_name)
        logger.info(f"Created batch with ID: {batch_id}, name: {batch_name}")
    except Exception as e:
        logger.error(f"Failed to create batch in database: {str(e)}")
        return {
            "status_code": 500,
            "message": f"Failed to create batch in database: {str(e)}",
            "total_check_amount": 0,
            "successful_docs": 0,
            "exception_docs": 0
        }
    
    # Process each uploaded file
    for file_obj in input_files:
        filename = file_obj.filename
        if not filename.lower().endswith((".tif", ".tiff", ".pdf")):
            logger.warning(f"Skipping {filename}: Not a valid TIFF or PDF file")
            continue
        file_url = ""
        
        logger.info(f"Processing {filename}...")
        print(f"Processing {filename}...")
        
        # First save file object to temporary file for processing
        temp_file_path = save_fileobj_to_temp_and_process(file_obj, filename)
        if not temp_file_path:
            exception_reason = "Failed to save file object to temporary file"
            
            # Create document entry with "exception" status and error JSON
            try:
                doc_id = db_ops["insert_document_in_db"](
                    batch_id=batch_id,
                    doc_name=filename,
                    type_id=category_ids_map.get("unknown"),
                    status="exception",
                    file_url=file_url,
                    json_content=json.dumps({"error": exception_reason})
                )
                
                # Add exception record
                db_ops["add_exception_in_db"](
                    document_id=doc_id,
                    exception_message=exception_reason
                )
                
                exceptions_list.append({"filename": filename, "reason": exception_reason})
                exception_docs += 1
            except Exception as e:
                logger.error(f"Database error for {filename}: {str(e)}")
            
            continue
        
        # Upload the file to S3 RAW Data directory
        raw_data_s3_path = os.path.join(base_dir, "RAW Data", filename)
        try:
            file_url = upload_file_to_s3(temp_file_path, raw_data_s3_path)
            logger.info(f"Uploaded {filename} to {raw_data_s3_path}")
        except Exception as e:
            logger.error(f"Failed to upload {filename} to S3: {str(e)}")
            exception_reason = f"Failed to upload file to S3: {str(e)}"
            
            # Create document entry with "exception" status and error JSON
            try:
                doc_id = db_ops["insert_document_in_db"](
                    batch_id=batch_id,
                    doc_name=filename,
                    type_id=category_ids_map.get("unknown"),
                    status="exception",
                    file_url=file_url,
                    json_content=json.dumps({"error": exception_reason})
                )
                
                # Add exception record
                db_ops["add_exception_in_db"](
                    document_id=doc_id,
                    exception_message=exception_reason
                )
                
                exceptions_list.append({"filename": filename, "reason": exception_reason})
                exception_docs += 1
            except Exception as db_e:
                logger.error(f"Database error for {filename}: {str(db_e)}")
            
            # Clean up temp file 
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
                
            continue
        
        # Check if this is a "No Money - Cancel" document using image matching
        try:
            if os.path.exists(REFERENCE_FILE) and is_similar_to_reference(temp_file_path):
                logger.info(f"Document {filename} identified as No Money - Cancel")
                category = "No Money - Cancel"
                
                # Create JSON data for the document
                json_data = {"filename": filename, "category": category, "Important_Info": {}}
                json_str = json.dumps(json_data, indent=4)
                
                # Move file to appropriate category directory in S3
                dest_file_dir = os.path.join(base_dir, category, "Passed", "Files")
                dest_s3_path = os.path.join(dest_file_dir, filename)
                file_url = safe_copy_file_to_s3(temp_file_path, dest_s3_path)
                # Create document entry
                try:
                    doc_id = db_ops["insert_document_in_db"](
                        batch_id=batch_id,
                        doc_name=filename,
                        type_id=category_ids_map.get(category.lower()),
                        status="processed",
                        file_url=file_url,
                        json_content=json_str
                    )
                    successful_docs += 1
                except Exception as e:
                    logger.error(f"Database error for {filename}: {str(e)}")
                    exception_reason = f"Database error: {str(e)}"
                    exceptions_list.append({"filename": filename, "reason": exception_reason})
                    exception_docs += 1
                
                
                # Clean up temp file and continue to the next document
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
                
                continue
        except Exception as e:
            logger.error(f"Error checking for No Money - Cancel: {str(e)}")
            # Continue with normal processing
        
        # Extract text from the file
        file_ext = os.path.splitext(filename)[1].lower()
        file_obj.seek(0)  # Reset file pointer to beginning
        text = extract_text_from_fileobj(file_obj, file_ext)
        
        if not text:
            logger.warning(f"No text extracted from {filename}")
            exception_reason = "No text could be extracted from file"
            
            # Create document entry with "exception" status and empty JSON
            try:
                doc_id = db_ops["insert_document_in_db"](
                    batch_id=batch_id,
                    doc_name=filename,
                    type_id=category_ids_map.get("unknown"),
                    status="exception",
                    file_url=file_url,
                    json_content=json.dumps({"error": exception_reason})
                )
                
                # Add exception record
                db_ops["add_exception_in_db"](
                    document_id=doc_id,
                    exception_message=exception_reason
                )
                
                exceptions_list.append({"filename": filename, "reason": exception_reason})
                exception_docs += 1
            except Exception as e:
                logger.error(f"Database error for {filename}: {str(e)}")
            
            # Move file to unknown category
            dest_file_dir = os.path.join(base_dir, "unknown", "Files")
            dest_s3_path = os.path.join(dest_file_dir, filename)
            file_url = safe_copy_file_to_s3(temp_file_path, dest_s3_path)
            
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
            
            continue
        
        # Classify the document
        category = classify_document(text)
        json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        temp_json_path = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(filename)[0]}.json")
        
        if category.lower() != "no money - cancel":
            try:
                client = openai.OpenAI(api_key=api_key)
                essentials = [k for k, v in category_variations.get(category.lower(), {}).items() if v["essential"]]
                prompt = (
                    f"Intelligently extract all possible meaningful key-value pairs from this {category} document as JSON. "
                    f"Ensure these essential keys are included in a *top-level* key named 'importantInfo' (in camelCase with no underscores). "
                    f"The essential keys are: {', '.join(essentials)}. If a key is not found, assign it an empty string. "
                    f"All other relevant non-essential information must go into a separate *top-level* key named 'extractedInfo'. "
                    f"Only these two keys—'importantInfo' and 'extractedInfo'—should exist at the top level, along with 'filename' and 'category'. "
                    f"Capture all relevant details such as names, numbers, dates, addresses, amounts, etc., that fit the {category} context. "
                    "Ensure policy and loan numbers are treated as distinct values, even if presented together. "
                    "If the document contains multiple policy numbers, addresses, and loan numbers (e.g., in rows or columns), capture them as arrays. "
                    "Strictly use camelCase for all keys with no special characters or spaces. "
                    "Do not nest 'importantInfo' or 'extractedInfo' under any other key or add duplicates. "
                    f"Document: {text[:2000]}\n\nJSON:"
                )
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=1500
                )
                key_values = json.loads(response.choices[0].message.content)
                
                variations = category_variations.get(category.lower(), {})
                extracted = search_key(key_values, list(variations.keys()), {k: v["variations"] for k, v in variations.items()})
                essentials_dict = {k: "" for k, v in variations.items() if v["essential"]}
                key_values = {**essentials_dict, **key_values, **extracted}
                key_values = convert_keys_to_camel_case(key_values)
                
                json_data = {"filename": filename, "category": category, "importantInfo": key_values}
                json_str = json.dumps(json_data, indent=4, cls=DecimalEncoder)
                
                try:
                    with open(temp_json_path, "w") as f:
                        f.write(json_str)
                    
                    # Upload JSON to S3
                    upload_file_to_s3(temp_json_path, json_path)
                    logger.info(f"Uploaded JSON to S3 at {json_path}")
                    
                    json_tif_pairs.append((temp_json_path, temp_file_path, json_str))
                    
                    if category.lower() == "check":
                        amount = extracted.get("amount", Decimal('0'))
                        if amount:
                            try:
                                formatted_amount = Decimal(amount).quantize(Decimal('0.01'))
                                print(f"Check amount from {filename}: ${formatted_amount}\n")
                                check_amounts.append(Decimal(amount))
                            except (InvalidOperation, TypeError) as e:
                                logger.warning(f"Could not format check amount for {filename}: {str(e)}")
                                exception_reason = f"Invalid check amount format: {str(e)}"
                                
                                # Create document entry with appropriate status and JSON
                                doc_id = db_ops["insert_document_in_db"](
                                    batch_id=batch_id,
                                    doc_name=filename,
                                    type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
                                    status="exception",
                                    file_url=file_url,
                                    json_content=json_str
                                )
                                
                                # Add exception record
                                db_ops["add_exception_in_db"](
                                    document_id=doc_id,
                                    exception_message=exception_reason
                                )
                                
                                exceptions_list.append({"filename": filename, "reason": exception_reason})
                                exception_docs += 1
                except (IOError, OSError) as e:
                    logger.error(f"Failed to write JSON file {temp_json_path}: {str(e)}")
                    exception_reason = f"Failed to write JSON file: {str(e)}"
                    
                    # Create document entry with appropriate status and JSON
                    doc_id = db_ops["insert_document_in_db"](
                        batch_id=batch_id,
                        doc_name=filename,
                        type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
                        status="exception",
                        file_url=file_url,
                        json_content=json_str
                    )
                    
                    # Add exception record
                    db_ops["add_exception_in_db"](
                        document_id=doc_id,
                        exception_message=exception_reason
                    )
                    
                    exceptions_list.append({"filename": filename, "reason": exception_reason})
                    exception_docs += 1
                
            except Exception as e:
                logger.error(f"Failed to process {filename} for key-value extraction: {str(e)}")
                exception_reason = f"Key-value extraction failed: {str(e)}"
                
                # Create document entry with appropriate status and error JSON
                error_json = json.dumps({"error": exception_reason, "filename": filename, "category": category})
                # Move file to appropriate failed category
                dest_file_dir = os.path.join(base_dir, category.capitalize(), "Failed", "Files")
                dest_s3_path = os.path.join(dest_file_dir, filename)
                file_url = safe_copy_file_to_s3(temp_file_path, dest_s3_path)
                doc_id = db_ops["insert_document_in_db"](
                    batch_id=batch_id,
                    doc_name=filename,
                    type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
                    status="exception",
                    file_url=file_url,
                    json_content=error_json
                )
                
                # Add exception record
                db_ops["add_exception_in_db"](
                    document_id=doc_id,
                    exception_message=exception_reason
                )
                
                exceptions_list.append({"filename": filename, "reason": exception_reason})
                exception_docs += 1
                
                
                # Clean up temp files
                try:
                    os.unlink(temp_file_path)
                    if os.path.exists(temp_json_path):
                        os.unlink(temp_json_path)
                except Exception:
                    pass
                
                continue
        else:
            # No Money - Cancel category
            json_data = {"filename": filename, "category": "no money - cancel", "importantInfo": {}}
            json_str = json.dumps(json_data, indent=4)
            
            # Create document entry
            try:
                doc_id = db_ops["insert_document_in_db"](
                    batch_id=batch_id,
                    doc_name=filename,
                    type_id=category_ids_map.get("no money - cancel"),
                    status="processed",
                    file_url=file_url,
                    json_content=json_str
                )
                successful_docs += 1
            except Exception as e:
                logger.error(f"Database error for {filename}: {str(e)}")
                exception_reason = f"Database error: {str(e)}"
                exceptions_list.append({"filename": filename, "reason": exception_reason})
                exception_docs += 1
            
            # Write JSON to temp file
            with open(temp_json_path, "w") as f:
                f.write(json_str)
            
            # Upload JSON to S3
            json_s3_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            upload_file_to_s3(temp_json_path, json_s3_path)
            
            # Move file to appropriate category directory
            dest_file_dir = os.path.join(base_dir, "no money - cancel", "Passed", "Files")
            dest_s3_path = os.path.join(dest_file_dir, filename)
            file_url = safe_copy_file_to_s3(temp_file_path, dest_s3_path)
            
            # Clean up temp files
            try:
                os.unlink(temp_file_path)
                os.unlink(temp_json_path)
            except Exception:
                pass
            
            json_tif_pairs.append((None, temp_file_path, json_str))
    
    total_check_amount = Decimal('0')
    if check_amounts:
        try:
            total_check_amount = sum(check_amounts)
            formatted_total = total_check_amount.quantize(Decimal('0.01'))
            print(f"Total sum of all check amounts: ${formatted_total}")
        except Exception as e:
            logger.error(f"Failed to calculate total check amount: {str(e)}")
    
    for json_path, tif_path, json_str in json_tif_pairs:
        if json_path:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                category = data["category"].lower()
                cat_dir = os.path.join(base_dir, category.capitalize() if category != "unknown" else category)
                variations = category_variations.get(category, {})
                
                # Extract important info for validation
                results = {}
                if "importantInfo" in data:
                    importantInfo = data["importantInfo"]
                    results = search_key(importantInfo, list(variations.keys()), {k: v["variations"] for k, v in variations.items()}) if variations else {}
                else:
                    for key, val in data.items():
                        if key.lower() in [k.lower() for k in variations.keys()]:
                            results[key] = val
                
                passed = False
                reasons = []
                
                if category == "check":
                    if not results.get("policyNumber", ""):
                        reasons.append("Missing policyNumber")
                    if not results.get("loanNumber", ""):
                        reasons.append("Missing loanNumber")
                    if results.get("amount", "0") == "0":
                        reasons.append("Invalid amount (zero or missing)")
                    passed = not reasons
                elif category == "mortgage":
                    required_keys = ["policyNumber", "name", "address"]
                    for key in required_keys:
                        if not results.get(key, ""):
                            reasons.append(f"Missing {key}")
                    passed = not reasons
                elif category == "claim":
                    required_keys = ["claimNumber", "name", "address"]
                    for key in required_keys:
                        if not results.get(key, ""):
                            reasons.append(f"Missing {key}")
                    passed = not reasons
                elif category == "coupons":
                    if not results.get("policyNumber", ""):
                        reasons.append("Missing policyNumber")
                    if not results.get("insured", ""):
                        reasons.append("Missing insured")
                    if results.get("amount", "0") == "0":
                        reasons.append("Invalid amount (zero or missing)")
                    passed = not reasons
                elif category == "agency":
                    if not (results.get("agencyName", "") or results.get("producer", "")):
                        reasons.append("Missing both agencyName and producer")
                    passed = not reasons
                
                # Insert document into database
                try:
                    status = "processed" if passed else "exception"
                    doc_id = db_ops["insert_document_in_db"](
                        batch_id=batch_id,
                        doc_name=os.path.basename(tif_path),
                        type_id=category_ids_map.get(category, category_ids_map.get("unknown")),
                        status=status,
                        file_url=file_url,
                        json_content=json_str
                    )
                    
                    # Insert document-specific information into the appropriate table
                    if passed:
                        # Insert document details into the appropriate type-specific table
                        try:
                            if category == "check":
                                policyNumber_raw = results.get("policyNumber", "")
                                try:
                                    parsed_policy = ast.literal_eval(policyNumber_raw)
                                    if isinstance(parsed_policy, list) and parsed_policy:
                                        policyNumber = parsed_policy[0]
                                    else:
                                        policyNumber = policyNumber_raw
                                except (ValueError, SyntaxError):
                                    policyNumber = policyNumber_raw
                                
                                db_ops["insert_check_document"](
                                    document_id=doc_id,
                                    policynumber=policyNumber,
                                    loannumber=results.get("loanNumber", ""),
                                    amount=Decimal(results.get("amount", "0"))
                                )
                            elif category == "mortgage":
                                db_ops["insert_mortgage_document"](
                                    document_id=doc_id,
                                    policynumber=results.get("policyNumber", ""),
                                    name=results.get("name", ""),
                                    address=results.get("address", "")
                                )
                            elif category == "claim":
                                db_ops["insert_claim_document"](
                                    document_id=doc_id,
                                    claimnumber=results.get("claimNumber", ""),
                                    name=results.get("name", ""),
                                    address=results.get("address", "")
                                )
                            elif category == "coupons":
                                db_ops["insert_coupon_document"](
                                    document_id=doc_id,
                                    policynumber=results.get("policyNumber", ""),
                                    insured=results.get("insured", ""),
                                    amount=Decimal(results.get("amount", "0"))
                                )
                            elif category == "agency":
                                db_ops["insert_agency_document"](
                                    document_id=doc_id,
                                    agencyname=results.get("agencyName", ""),
                                    producer=results.get("producer", "")
                                )
                            logger.info(f"Successfully inserted document details for {os.path.basename(tif_path)} into {category} table")
                        except Exception as e:
                            logger.error(f"Failed to insert details into {category} table for document {doc_id}: {str(e)}")
                            # Add an exception for this specific error
                            db_ops["add_exception_in_db"](
                                document_id=doc_id,
                                exception_message=f"Failed to insert details into {category} table: {str(e)}"
                            )
                    
                    # If there are failure reasons, add them as exceptions
                    if not passed:
                        exception_reason = ", ".join(reasons)
                        db_ops["add_exception_in_db"](
                            document_id=doc_id,
                            exception_message=exception_reason
                        )
                        exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
                        exception_docs += 1
                        print(f"File {os.path.basename(tif_path)} failed for category '{category}' due to: {exception_reason}")
                    else:
                        successful_docs += 1
                        
                except Exception as e:
                    logger.error(f"Database error for {os.path.basename(tif_path)}: {str(e)}")
                    exception_reason = f"Database error: {str(e)}"
                    # Move files to appropriate category directories in S3
                    dest_json_dir = os.path.join(cat_dir, "Passed" if passed else "Failed", "JSON")
                    dest_file_dir = os.path.join(cat_dir, "Passed" if passed else "Failed", "Files")
                    
                    # Move the JSON file
                    json_s3_path = os.path.join(dest_json_dir, os.path.basename(json_path))
                    safe_copy_file_to_s3(json_path, json_s3_path)
                    
                    # Move the TIFF/PDF file
                    file_s3_path = os.path.join(dest_file_dir, os.path.basename(tif_path))
                    file_url = safe_copy_file_to_s3(tif_path, file_s3_path)
                    
                    # Try to create a document entry with exception status
                    try:
                        doc_id = db_ops["insert_document_in_db"](
                            batch_id=batch_id,
                            doc_name=os.path.basename(tif_path),
                            type_id=category_ids_map.get(category, category_ids_map.get("unknown")),
                            status="exception",
                            file_url=file_url,
                            json_content=json_str
                        )
                        
                        db_ops["add_exception_in_db"](
                            document_id=doc_id,
                            exception_message=exception_reason
                        )
                    except Exception as inner_e:
                        logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                    
                    exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
                    exception_docs += 1
                
                
                # Clean up temp files
                try:
                    os.unlink(json_path)
                    os.unlink(tif_path)
                except Exception:
                    pass
                
            except Exception as e:
                logger.error(f"Failed to process {os.path.basename(json_path)}: {str(e)}")
                cat_dir = os.path.join(base_dir, "unknown")
                
                # Create document entry with "exception" status and error JSON
                error_json = json.dumps({"error": f"Processing error: {str(e)}", "filename": os.path.basename(tif_path)})
                file_s3_path = os.path.join(cat_dir, os.path.basename(tif_path))
                file_url = safe_copy_file_to_s3(tif_path, file_s3_path)
                try:
                    doc_id = db_ops["insert_document_in_db"](
                        batch_id=batch_id,
                        doc_name=os.path.basename(tif_path),
                        type_id=category_ids_map.get("unknown"),
                        status="exception",
                        file_url=file_url,
                        json_content=error_json
                    )
                    
                    db_ops["add_exception_in_db"](
                        document_id=doc_id,
                        exception_message=f"Processing error: {str(e)}"
                    )
                except Exception as inner_e:
                    logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                
                # Move file to unknown category
                
                # Clean up temp files
                try:
                    os.unlink(json_path)
                    os.unlink(tif_path)
                except Exception:
                    pass
                
                exceptions_list.append({"filename": os.path.basename(tif_path), "reason": f"Processing error: {str(e)}"})
                exception_docs += 1
                print(f"File {os.path.basename(tif_path)} moved to 'unknown' due to processing error: {str(e)}")
    
    logger.info("Processing complete.")
    
    # Return the requested information
    return {
        "status_code": 200,
        "message": "Document processing completed successfully",
        "total_check_amount": str(total_check_amount.quantize(Decimal('0.01'))),
        "successful_docs": successful_docs,
        "exception_docs": exception_docs,
        "exceptions": exceptions_list  # Optional: return the list of exceptions if needed
    }

if __name__ == "__main__":
    try:
        # For testing purposes only - would be replaced by actual file upload handling in production
        class MockFile:
            def __init__(self, filename, path):
                self.filename = filename
                self.path = path
                self._file = open(path, 'rb')
            
            def read(self):
                self._file.seek(0)
                return self._file.read()
            
            def seek(self, pos):
                return self._file.seek(pos)
            
            def __del__(self):
                try:
                    self._file.close()
                except:
                    pass
        
        # Test with some files in the RAW Data directory
        import glob
        test_files = []
        for file_path in glob.glob(os.path.join("Files", "RAW Data", "*")):
            if file_path.lower().endswith((".tif", ".tiff", ".pdf")):
                test_files.append(MockFile(os.path.basename(file_path), file_path))
        
        if test_files:
            success = process_files(test_files)
            if success:
                print("Processing completed successfully.")
            else:
                print("Processing finished with errors. Check log for details.")
        else:
            print("No test files found. Please place files in Files/RAW Data directory.")
    except Exception as e:
        logger.critical(f"Fatal error during processing: {str(e)}")
        print("Processing failed with critical error. See log for details.")