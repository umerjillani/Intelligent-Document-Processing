import os
import re
import json
import shutil
import logging
from decimal import Decimal, InvalidOperation, getcontext, ROUND_HALF_UP
import importlib.util
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


from dbOps import ensure_document_apitypes_in_db, create_batch_in_db, insert_document_in_db, add_exception_in_db, get_db_connection, insert_agency_document, insert_coupon_document, insert_claim_document, insert_mortgage_document, insert_check_document

print('1')

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
print('2')

# Check required imports
REQUIRED_PACKAGES = [
    "PIL", "pytesseract", "fitz", "transformers", "openai", "cv2"
]

missing_packages = []
for package in REQUIRED_PACKAGES:
    if importlib.util.find_spec(package) is None:
        missing_packages.append(package)

print('3')
if missing_packages:
    error_msg = f"Missing required packages: {', '.join(missing_packages)}. Please install them using pip."
    logger.error(error_msg)
    raise ImportError(error_msg)

print('4')
# Set precision for Decimal calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

try:
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large", use_fast=False)
    # model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-large")
    # classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    
    # tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large", use_fast=False)
    # classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-large", tokenizer=tokenizer)
    # classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", from_flax=True)
    # classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-large", tokenizer=tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large")
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "cross-encoder/nli-deberta-v3-large",
    #     device_map="auto" if torch.cuda.is_available() else None,
    #     _fast_init=False
    # )
    # Then create the pipeline with the loaded model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
except Exception as e:
    logger.error(f"Failed to load NLP models: {str(e)}")
    raise

print('5')
CATEGORY_LABELS = ["Claim", "Mortgage", "No Money - Cancel", "Check", "Coupons", "Agency"]

category_variations = {
    "check": {
        "policynumber": {"variations": {"policy", "policynumber", "policyno"}, "essential": True},
        "loannumber": {"variations": {"loan", "loannumber", "loanno"}, "essential": True},
        "amount": {"variations": {"amount", "amt", "checkamount", "chequeamount", "totalamount", "netamount", "net amt"}, "essential": True}
    },
    "mortgage": {
        "policynumber": {"variations": {"policy", "policynumber", "policyno"}, "essential": True},
        "name": {"variations": {"name", "client name", "client"}, "essential": True},
        "address": {"variations": {"address", "propertyaddress", "property"}, "essential": True}
    },
    "claim": {
        "claimnumber": {"variations": {"claimnumber", "claim", "claimno", "claimno."}, "essential": True},
        "name": {"variations": {"name", "client name", "client"}, "essential": True},
        "address": {"variations": {"address", "propertyaddress", "property"}, "essential": True}
    },
    "coupons": {
        "policynumber": {"variations": {"policynumber", "policy", "policynumberid", "policyno"}, "essential": True},
        "insured": {"variations": {"insured", "insuredname"}, "essential": True},
        "amount": {"variations": {"amount", "amt", "checkamount", "chequeamount", "totalamount", "netamount", "net amt"}, "essential": True}
    },
    "agency": {
        "agencyname": {"variations": {"agency", "agencyname", "insurance agency"}, "essential": True},
        "producer": {"variations": {"producer", "producer name", "agent"}, "essential": True}
    }
}

tiff_dir = os.path.join("Files", "RAW Data")
output_dir = os.path.join("Files", "JSONs")
os.makedirs(output_dir, exist_ok=True)

REFERENCE_FILE = "Files/no_money_reference.png"  # Explicitly a PNG

print('6')
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

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

# def extract_text_from_pdf(pdf_path):
#     try:
#         pdf_document = fitz.open(pdf_path)
#         text = ""
#         for page_num in range(pdf_document.page_count):
#             page = pdf_document[page_num]
#             page_text = page.get_text("text").strip()
#             text += page_text + "\n" if page_text else ""
#         text = clean_ocr_text(text)
#         if not text or len(text) < 50:
#             text = ""
#             for page_num in range(pdf_document.page_count):
#                 try:
#                     page = pdf_document[page_num]
#                     pix = page.get_pixmap(dpi=300)
#                     img_bytes = pix.tobytes()
#                     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#                     page_text = pytesseract.image_to_string(img)
#                     text += page_text + "\n"
#                 except Exception as e:
#                     logging.error(f"Error performing OCR on page {page_num} of PDF {pdf_path}: {str(e)}")
#                     continue
#             text = clean_ocr_text(text)
#         pdf_document.close()
#         return text
#     except Exception as e:
#         logging.error(f"PDF text extraction failed for {pdf_path}: {str(e)}")
#         return ""

def extract_text_from_pdf(file_obj):
    try:
        pdf_document = fitz.open(stream=file_obj.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text("text").strip()
            text += page_text + "\n" if page_text else ""
        text = clean_ocr_text(text)
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
                    logging.error(f"Error performing OCR on page {page_num} of PDF: {str(e)}")
                    continue
            text = clean_ocr_text(text)
        pdf_document.close()
        return text
    except Exception as e:
        logging.error(f"PDF text extraction failed: {str(e)}")
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
    elif any(phrase in text_lower for phrase in ["pay to the order of", "pay to the"]):
        return "Check"
    return None

def classify_document(text):
    quick_result = quick_classify(text)
    if quick_result:
        print(f"Classified by Quick_Classify as {quick_result}.\n")
        return quick_result
    
    text_to_classify = " ".join(text.split()[:2000])
    
    try:
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
            print(f"Classified by Zero-shot as {result['labels'][0]} with confidence = {result['scores'][0]}\n")
            return result['labels'][0]
    except Exception as e:
        logger.warning(f"Zero-shot classification failed: {str(e)}")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Cannot perform classification.")
        return "Unknown"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "Classify this document into one category:\n"
            "- Check: Payment checks with 'Pay to the order of' text, dollar amounts, check numbers, bank routing numbers, payee names, and signatures.\n"
            "- Coupons: Payment vouchers, premium notices, detachable bill payment stubs with amount due, payment instructions, policy numbers, and possibly tear-off sections.\n"
            "- No Money - Cancel: Cancelled transaction notices, policy terminations, void notifications, insufficient funds notices with cancellation dates, fee information, or refund details.\n"
            "- Claim: Insurance claim documents with distinct claim numbers, date of loss information, damage descriptions, claim status updates, adjuster details, and references to specific damage incidents or losses.\n"
            "- Agency: Insurance agency administrative documents containing agent licensing info, producer codes, agency agreements, commission statements, without specific claim numbers or loss details. References to agency operations rather than individual claims.\n"
            "- Mortgage: Property financing documents with loan numbers, principal amounts, interest rates, property addresses, lender details, deed information, and payment schedules.\n\n"
            f"Document: {text[:1000]}\n\nCategory:"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
        
        raw_category = response.choices[0].message.content.strip()
        valid_categories = {cat.lower(): cat for cat in CATEGORY_LABELS}
        normalized_response = raw_category.lower()
        
        if normalized_response in valid_categories:
            category = valid_categories[normalized_response]
            print(f"Classified by OpenAI as {category}")
            return category
        
        for valid_key in valid_categories:
            if valid_key in normalized_response:
                category = valid_categories[valid_key]
                print(f"Classified by OpenAI as {category} (from partial match: '{raw_category}')")
                return category
        
        print(f"OpenAI returned unrecognized category: '{raw_category}', defaulting to 'Unknown'")
        return "Unknown"
    except Exception as e:
        logger.error(f"OpenAI classification failed: {str(e)}")
        return "Unknown"

def normalize_string(value):
    if not value: return ""
    if not isinstance(value, str): value = str(value)
    return re.sub(r'[^a-zA-Z0-9]', '', value).lower().strip()

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
                        if target in [normalize_string(k) for k in ["amount", "checkamount", "totalamount", "netamount"]]:
                            try:
                                cleaned_value = str(value).replace('$', '').replace(',', '').strip()
                                amount = Decimal(cleaned_value)
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

def safe_copy_file(src, dest):
    try:
        shutil.copy(src, dest)
        return True
    except (IOError, OSError) as e:
        logger.error(f"Failed to copy file from {src} to {dest}: {str(e)}")
        return False

def safe_copy_file_obj(file_obj, dest):
    try:
        with open(dest, 'wb') as tiff_file:
            shutil.copyfileobj(file_obj, tiff_file)
        return True
    except (IOError, OSError) as e:
        logger.error(f"Failed to copy document to {dest}: {str(e)}")
        return False

# def process_files(input_dir, output_dir):
#     if not api_key:
#         logger.error("API key not found. Cannot proceed with document processing.")
#         return False
    
#     base_dir = "Files"
#     categories = ["check", "mortgage", "claim", "coupons", "agency", "no money - cancel", "unknown"]
    
#     for cat in categories:
#         for status in ["passed", "failed"]:
#             os.makedirs(os.path.join(base_dir, cat, status, "files"), exist_ok=True)
#             os.makedirs(os.path.join(base_dir, cat, status, "jsons"), exist_ok=True)
    
#     json_tif_pairs = []
#     check_amounts = []
    
#     # if not os.path.exists(input_dir):
#     #     logger.error(f"Input directory {input_dir} does not exist.")
#     #     return False
#     files_dir = os.path.join("Files", "RAW Data")
#     for fileObj in (input_dir):
#         filename = fileObj.filename
#         file_path = os.path.join(files_dir, filename)
#         if not filename.lower().endswith((".tif", ".tiff", ".pdf")):
#             continue
        
#         print(f"Processing {filename}...")
        
#         # if is_similar_to_reference(fileObj):
#         #     dest_file_dir = os.path.join(base_dir, "no money - cancel", "passed", "files")
#         #     safe_copy_file(file_path, os.path.join(dest_file_dir, os.path.basename(file_path)))
#         #     print(f"File {filename} matched reference 'No Money - Cancel' by image similarity (70%+ match) and moved.")
#         #     json_tif_pairs.append((None, file_path))
#         #     continue
        
#         text = extract_text_from_tiff(fileObj) if filename.lower().endswith((".tif", ".tiff")) else extract_text_from_pdf(fileObj)
#         if not text:
#             logger.warning(f"No text extracted from {filename}")
#             continue
        
#         category = classify_document(text)
#         json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        
#         if category.lower() != "no money - cancel":
#             try:
#                 client = openai.OpenAI(api_key=api_key)
#                 essentials = [k for k, v in category_variations.get(category.lower(), {}).items() if v["essential"]]
#                 prompt = (
#                     f"Extract all possible meaningful key-value pairs from this {category} document as JSON. "
#                     f"Ensure these essential keys are included (empty string if not found): {', '.join(essentials)}. "
#                     "Capture all relevant details such as names, numbers, dates, addresses, amounts, etc., that fit the {category} context. "
#                     "Make sure Policy & Loan numbers are distinct, they are written with each other but are different. "
#                     "Focus on accuracy and relevance, avoiding generic or unrelated pairs.\n\n"
#                     f"Document: {text[:2000]}\n\nJSON:"
#                 )
#                 response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=[{"role": "user", "content": prompt}],
#                     response_format={"type": "json_object"},
#                     max_tokens=1500
#                 )
#                 key_values = json.loads(response.choices[0].message.content)
                
#                 variations = category_variations.get(category.lower(), {})
#                 extracted = search_key(key_values, list(variations.keys()), {k: v["variations"] for k, v in variations.items()})
#                 essentials_dict = {k: "" for k, v in variations.items() if v["essential"]}
#                 key_values = {**essentials_dict, **key_values, **extracted}
                
#                 json_data = {"filename": filename, "category": category, "Important_Info": key_values}
#                 try:
#                     with open(json_path, "w") as f:
#                         json.dump(json_data, f, indent=4, cls=DecimalEncoder)
#                     json_tif_pairs.append((json_path, file_path))
                    
#                     if category.lower() == "check":
#                         amount = extracted.get("amount", Decimal('0'))
#                         if amount:
#                             try:
#                                 formatted_amount = Decimal(amount).quantize(Decimal('0.01'))
#                                 print(f"Check amount from {filename}: ${formatted_amount}\n")
#                                 check_amounts.append(Decimal(amount))
#                             except (InvalidOperation, TypeError) as e:
#                                 logger.warning(f"Could not format check amount for {filename}: {str(e)}")
#                 except (IOError, OSError) as e:
#                     logger.error(f"Failed to write JSON file {json_path}: {str(e)}")
                
#             except Exception as e:
#                 logger.error(f"Failed to process {filename} for key-value extraction: {str(e)}")
#                 continue
#         else:
#             json_tif_pairs.append((None, file_path))
    
#     if check_amounts:
#         try:
#             total_sum = sum(check_amounts)
#             formatted_total = total_sum.quantize(Decimal('0.01'))
#             print(f"Total sum of all check amounts: ${formatted_total}")
#         except Exception as e:
#             logger.error(f"Failed to calculate total check amount: {str(e)}")
    
#     for json_path, tif_path in json_tif_pairs:
#         if json_path:
#             try:
#                 with open(json_path, 'r') as f:
#                     data = json.load(f)

#                 category = data["category"].lower()
#                 cat_dir = os.path.join(base_dir, category)
#                 variations = category_variations.get(category, {})
#                 results = search_key(data["Important_Info"], list(variations.keys()), {k: v["variations"] for k, v in variations.items()}) if variations else {}
                
#                 passed = False
#                 reasons = []
                
#                 if category == "check":
#                     if not results.get("policynumber", ""):
#                         reasons.append("Missing policynumber")
#                     if results.get("amount", "0") == "0":
#                         reasons.append("Invalid amount (zero or missing)")
#                     passed = not reasons
#                 elif category == "mortgage":
#                     required_keys = ["policynumber", "name", "address"]
#                     for key in required_keys:
#                         if not results.get(key, ""):
#                             reasons.append(f"Missing {key}")
#                     passed = not reasons
#                 elif category == "claim":
#                     required_keys = ["claimnumber", "name", "address"]
#                     for key in required_keys:
#                         if not results.get(key, ""):
#                             reasons.append(f"Missing {key}")
#                     passed = not reasons
#                 elif category == "coupons":
#                     if not results.get("policynumber", ""):
#                         reasons.append("Missing policynumber")
#                     if not results.get("insured", ""):
#                         reasons.append("Missing insured")
#                     if results.get("amount", "0") == "0":
#                         reasons.append("Invalid amount (zero or missing)")
#                     passed = not reasons
#                 elif category == "agency":
#                     if not (results.get("agencyname", "") or results.get("producer", "")):
#                         reasons.append("Missing both agencyname and producer")
#                     passed = not reasons
                
#                 if not passed:
#                     print(f"File {os.path.basename(tif_path)} failed for category '{category}' due to: {', '.join(reasons)}")
                
#                 dest_json_dir = os.path.join(cat_dir, "passed" if passed else "failed", "jsons")
#                 dest_file_dir = os.path.join(cat_dir, "passed" if passed else "failed", "files")
                
#                 json_copied = safe_copy_file(json_path, os.path.join(dest_json_dir, os.path.basename(json_path)))
#                 file_copied = safe_copy_file_obj(fileObj, os.path.join(dest_file_dir, os.path.basename(tif_path)))
                
#                 if not json_copied or not file_copied:
#                     logger.warning(f"Failed to move some files for {os.path.basename(json_path)}")
#             except Exception as e:
#                 logger.error(f"Failed to process {os.path.basename(json_path)}: {str(e)}")
#                 cat_dir = os.path.join(base_dir, "unknown")
#                 safe_copy_file(json_path, os.path.join(cat_dir, "failed", "jsons", os.path.basename(json_path)))
#                 safe_copy_file(tif_path, os.path.join(cat_dir, "failed", "files", os.path.basename(tif_path)))
#                 print(f"File {os.path.basename(tif_path)} moved to 'unknown/failed' due to processing error: {str(e)}")
#         else:
#             dest_file_dir = os.path.join(base_dir, "no money - cancel", "passed", "files")
#             os.makedirs(dest_file_dir, exist_ok=True)
#             if not safe_copy_file(tif_path, os.path.join(dest_file_dir, os.path.basename(tif_path))):
#                 logger.error(f"Failed to move {os.path.basename(tif_path)} to no money - cancel directory")
#             else:
#                 print(f"File {os.path.basename(tif_path)} moved to 'no money - cancel/passed' as it was classified as 'No Money - Cancel'")
#         print("\n")
#     logger.info("Processing complete.")
#     return True

# def process_files(input_dir, output_dir):
#     """
#     Process files from input directory, categorize them, extract information,
#     and store results in the database according to the provided schema.
    
#     Returns:
#         dict: Result containing status code, message, and statistics
#     """
#     if not api_key:
#         logger.error("API key not found. Cannot proceed with document processing.")
#         return {
#             "status_code": 400,
#             "message": "API key not found. Cannot proceed with document processing.",
#             "total_check_amount": 0,
#             "successful_docs": 0,
#             "exception_docs": 0
#         }
    
#     # Initialize counters and tracking variables
#     successful_docs = 0
#     exception_docs = 0
#     check_amounts = []
#     exceptions_list = []
    
#     base_dir = "Files"
#     categories = ["check", "mortgage", "claim", "coupons", "agency", "no money - cancel", "unknown"]
    
#     # Ensure document type IDs are in the database
#     try:
#         category_ids_map = ensure_document_types_in_db(categories)
#     except Exception as e:
#         logger.error(f"Failed to ensure document types in database: {str(e)}")
#         return {
#             "status_code": 500,
#             "message": f"Failed to prepare document types: {str(e)}",
#             "total_check_amount": 0,
#             "successful_docs": 0,
#             "exception_docs": 0
#         }
    
#     for cat in categories:
#         for status in ["passed", "failed"]:
#             os.makedirs(os.path.join(base_dir, cat, status, "files"), exist_ok=True)
#             os.makedirs(os.path.join(base_dir, cat, status, "jsons"), exist_ok=True)
    
#     json_tif_pairs = []
    
#     # Generate a unique batch name based on timestamp
#     batch_name = f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
#     # Create a batch in the database
#     try:
#         batch_id = create_batch_in_db(batch_name)
#         logger.info(f"Created batch with ID: {batch_id}, name: {batch_name}")
#     except Exception as e:
#         logger.error(f"Failed to create batch in database: {str(e)}")
#         return {
#             "status_code": 500,
#             "message": f"Failed to create batch in database: {str(e)}",
#             "total_check_amount": 0,
#             "successful_docs": 0,
#             "exception_docs": 0
#         }
    
#     files_dir = os.path.join("Files", "RAW Data")
#     for fileObj in (input_dir):
#         filename = fileObj.filename
#         file_path = os.path.join(files_dir, filename)
#         if not filename.lower().endswith((".tif", ".tiff", ".pdf")):
#             continue
        
#         print(f"Processing {filename}...")
        
#         text = extract_text_from_tiff(fileObj) if filename.lower().endswith((".tif", ".tiff")) else extract_text_from_pdf(fileObj)
#         if not text:
#             logger.warning(f"No text extracted from {filename}")
#             exception_reason = "No text could be extracted from file"
            
#             # Create document entry with "exception" status
#             try:
#                 doc_id = insert_document_in_db(
#                     batch_id=batch_id,
#                     doc_name=filename,
#                     type_id=category_ids_map.get("unknown"),
#                     status="exception"
#                 )
                
#                 # Add exception record
#                 add_exception_in_db(
#                     document_id=doc_id,
#                     exception_message=exception_reason
#                 )
                
#                 exceptions_list.append({"filename": filename, "reason": exception_reason})
#                 exception_docs += 1
#             except Exception as e:
#                 logger.error(f"Database error for {filename}: {str(e)}")
            
#             continue
        
#         category = classify_document(text)
#         json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        
#         if category.lower() != "no money - cancel":
#             try:
#                 client = openai.OpenAI(api_key=api_key)
#                 essentials = [k for k, v in category_variations.get(category.lower(), {}).items() if v["essential"]]
#                 prompt = (
#                     f"Extract all possible meaningful key-value pairs from this {category} document as JSON. "
#                     f"Ensure these essential keys are included (empty string if not found): {', '.join(essentials)}. "
#                     "Capture all relevant details such as names, numbers, dates, addresses, amounts, etc., that fit the {category} context. "
#                     "Make sure Policy & Loan numbers are distinct, they are written with each other but are different. "
#                     "Focus on accuracy and relevance, avoiding generic or unrelated pairs.\n\n"
#                     f"Document: {text[:2000]}\n\nJSON:"
#                 )
#                 response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=[{"role": "user", "content": prompt}],
#                     response_format={"type": "json_object"},
#                     max_tokens=1500
#                 )
#                 key_values = json.loads(response.choices[0].message.content)
                
#                 variations = category_variations.get(category.lower(), {})
#                 extracted = search_key(key_values, list(variations.keys()), {k: v["variations"] for k, v in variations.items()})
#                 essentials_dict = {k: "" for k, v in variations.items() if v["essential"]}
#                 key_values = {**essentials_dict, **key_values, **extracted}
                
#                 json_data = {"filename": filename, "category": category, "Important_Info": key_values}
#                 try:
#                     with open(json_path, "w") as f:
#                         json.dump(json_data, f, indent=4, cls=DecimalEncoder)
#                     json_tif_pairs.append((json_path, file_path))
                    
#                     if category.lower() == "check":
#                         amount = extracted.get("amount", Decimal('0'))
#                         if amount:
#                             try:
#                                 formatted_amount = Decimal(amount).quantize(Decimal('0.01'))
#                                 print(f"Check amount from {filename}: ${formatted_amount}\n")
#                                 check_amounts.append(Decimal(amount))
#                             except (InvalidOperation, TypeError) as e:
#                                 logger.warning(f"Could not format check amount for {filename}: {str(e)}")
#                                 exception_reason = f"Invalid check amount format: {str(e)}"
                                
#                                 # Create document entry with appropriate status
#                                 doc_id = insert_document_in_db(
#                                     batch_id=batch_id,
#                                     doc_name=filename,
#                                     type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
#                                     status="exception"
#                                 )
                                
#                                 # Add exception record
#                                 add_exception_in_db(
#                                     document_id=doc_id,
#                                     exception_message=exception_reason
#                                 )
                                
#                                 exceptions_list.append({"filename": filename, "reason": exception_reason})
#                                 exception_docs += 1
#                 except (IOError, OSError) as e:
#                     logger.error(f"Failed to write JSON file {json_path}: {str(e)}")
#                     exception_reason = f"Failed to write JSON file: {str(e)}"
                    
#                     # Create document entry with appropriate status
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=filename,
#                         type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
#                         status="exception"
#                     )
                    
#                     # Add exception record
#                     add_exception_in_db(
#                         document_id=doc_id,
#                         exception_message=exception_reason
#                     )
                    
#                     exceptions_list.append({"filename": filename, "reason": exception_reason})
#                     exception_docs += 1
                
#             except Exception as e:
#                 logger.error(f"Failed to process {filename} for key-value extraction: {str(e)}")
#                 exception_reason = f"Key-value extraction failed: {str(e)}"
                
#                 # Create document entry with appropriate status
#                 doc_id = insert_document_in_db(
#                     batch_id=batch_id,
#                     doc_name=filename,
#                     type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
#                     status="exception"
#                 )
                
#                 # Add exception record
#                 add_exception_in_db(
#                     document_id=doc_id,
#                     exception_message=exception_reason
#                 )
                
#                 exceptions_list.append({"filename": filename, "reason": exception_reason})
#                 exception_docs += 1
#                 continue
#         else:
#             json_tif_pairs.append((None, file_path))
    
#     total_check_amount = Decimal('0')
#     if check_amounts:
#         try:
#             total_check_amount = sum(check_amounts)
#             formatted_total = total_check_amount.quantize(Decimal('0.01'))
#             print(f"Total sum of all check amounts: ${formatted_total}")
#         except Exception as e:
#             logger.error(f"Failed to calculate total check amount: {str(e)}")
    
#     for json_path, tif_path in json_tif_pairs:
#         if json_path:
#             try:
#                 with open(json_path, 'r') as f:
#                     data = json.load(f)

#                 category = data["category"].lower()
#                 cat_dir = os.path.join(base_dir, category)
#                 variations = category_variations.get(category, {})
#                 results = search_key(data["Important_Info"], list(variations.keys()), {k: v["variations"] for k, v in variations.items()}) if variations else {}
                
#                 passed = False
#                 reasons = []
                
#                 if category == "check":
#                     if not results.get("policynumber", ""):
#                         reasons.append("Missing policynumber")
#                     if results.get("amount", "0") == "0":
#                         reasons.append("Invalid amount (zero or missing)")
#                     passed = not reasons
#                 elif category == "mortgage":
#                     required_keys = ["policynumber", "name", "address"]
#                     for key in required_keys:
#                         if not results.get(key, ""):
#                             reasons.append(f"Missing {key}")
#                     passed = not reasons
#                 elif category == "claim":
#                     required_keys = ["claimnumber", "name", "address"]
#                     for key in required_keys:
#                         if not results.get(key, ""):
#                             reasons.append(f"Missing {key}")
#                     passed = not reasons
#                 elif category == "coupons":
#                     if not results.get("policynumber", ""):
#                         reasons.append("Missing policynumber")
#                     if not results.get("insured", ""):
#                         reasons.append("Missing insured")
#                     if results.get("amount", "0") == "0":
#                         reasons.append("Invalid amount (zero or missing)")
#                     passed = not reasons
#                 elif category == "agency":
#                     if not (results.get("agencyname", "") or results.get("producer", "")):
#                         reasons.append("Missing both agencyname and producer")
#                     passed = not reasons
                
#                 # Insert document into database
#                 try:
#                     status = "processed" if passed else "exception"
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=os.path.basename(tif_path),
#                         type_id=category_ids_map.get(category, category_ids_map.get("unknown")),
#                         status=status
#                     )
                    
#                     # If there are failure reasons, add them as exceptions
#                     if not passed:
#                         exception_reason = ", ".join(reasons)
#                         add_exception_in_db(
#                             document_id=doc_id,
#                             exception_message=exception_reason
#                         )
#                         exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                         exception_docs += 1
#                         print(f"File {os.path.basename(tif_path)} failed for category '{category}' due to: {exception_reason}")
#                     else:
#                         successful_docs += 1
                        
#                 except Exception as e:
#                     logger.error(f"Database error for {os.path.basename(tif_path)}: {str(e)}")
#                     exception_reason = f"Database error: {str(e)}"
                    
#                     # Try to create a document entry with exception status
#                     try:
#                         doc_id = insert_document_in_db(
#                             batch_id=batch_id,
#                             doc_name=os.path.basename(tif_path),
#                             type_id=category_ids_map.get(category, category_ids_map.get("unknown")),
#                             status="exception"
#                         )
                        
#                         add_exception_in_db(
#                             document_id=doc_id,
#                             exception_message=exception_reason
#                         )
#                     except Exception as inner_e:
#                         logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                    
#                     exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                     exception_docs += 1
                
#                 dest_json_dir = os.path.join(cat_dir, "passed" if passed else "failed", "jsons")
#                 dest_file_dir = os.path.join(cat_dir, "passed" if passed else "failed", "files")
                
#                 json_copied = safe_copy_file(json_path, os.path.join(dest_json_dir, os.path.basename(json_path)))
#                 file_copied = safe_copy_file_obj(fileObj, os.path.join(dest_file_dir, os.path.basename(tif_path)))
                
#                 if not json_copied or not file_copied:
#                     logger.warning(f"Failed to move some files for {os.path.basename(json_path)}")
                    
#             except Exception as e:
#                 logger.error(f"Failed to process {os.path.basename(json_path)}: {str(e)}")
#                 cat_dir = os.path.join(base_dir, "unknown")
#                 safe_copy_file(json_path, os.path.join(cat_dir, "failed", "jsons", os.path.basename(json_path)))
#                 safe_copy_file(tif_path, os.path.join(cat_dir, "failed", "files", os.path.basename(tif_path)))
#                 print(f"File {os.path.basename(tif_path)} moved to 'unknown/failed' due to processing error: {str(e)}")
                
#                 exception_reason = f"Processing error: {str(e)}"
                
#                 # Create document entry with "exception" status
#                 try:
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=os.path.basename(tif_path),
#                         type_id=category_ids_map.get("unknown"),
#                         status="exception"
#                     )
                    
#                     add_exception_in_db(
#                         document_id=doc_id,
#                         exception_message=exception_reason
#                     )
#                 except Exception as inner_e:
#                     logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                
#                 exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                 exception_docs += 1
#         else:
#             # No Money - Cancel category case
#             dest_file_dir = os.path.join(base_dir, "no money - cancel", "passed", "files")
#             os.makedirs(dest_file_dir, exist_ok=True)
#             if not safe_copy_file(tif_path, os.path.join(dest_file_dir, os.path.basename(tif_path))):
#                 logger.error(f"Failed to move {os.path.basename(tif_path)} to no money - cancel directory")
#                 exception_reason = "Failed to move file to no money - cancel directory"
                
#                 # Create document entry with "exception" status
#                 try:
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=os.path.basename(tif_path),
#                         type_id=category_ids_map.get("no money - cancel"),
#                         status="exception"
#                     )
                    
#                     add_exception_in_db(
#                         document_id=doc_id,
#                         exception_message=exception_reason
#                     )
#                 except Exception as inner_e:
#                     logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                
#                 exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                 exception_docs += 1
#             else:
#                 print(f"File {os.path.basename(tif_path)} moved to 'no money - cancel/passed' as it was classified as 'No Money - Cancel'")
                
#                 # Insert into database as a no-money document
#                 try:
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=os.path.basename(tif_path),
#                         type_id=category_ids_map.get("no money - cancel"),
#                         status="processed"
#                     )
#                     successful_docs += 1
#                 except Exception as e:
#                     logger.error(f"Database error for {os.path.basename(tif_path)}: {str(e)}")
#                     exception_reason = f"Database error: {str(e)}"
                    
#                     # Try to create a document entry with exception status
#                     try:
#                         doc_id = insert_document_in_db(
#                             batch_id=batch_id,
#                             doc_name=os.path.basename(tif_path),
#                             type_id=category_ids_map.get("no money - cancel"),
#                             status="exception"
#                         )
                        
#                         add_exception_in_db(
#                             document_id=doc_id,
#                             exception_message=exception_reason
#                         )
#                     except Exception as inner_e:
#                         logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                    
#                     exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                     exception_docs += 1
        
#         print("\n")
    
#     logger.info("Processing complete.")
    
#     # Return the requested information
#     return {
#         "status_code": 200,
#         "message": "Document processing completed successfully",
#         "total_check_amount": str(total_check_amount.quantize(Decimal('0.01'))),
#         "successful_docs": successful_docs,
#         "exception_docs": exception_docs,
#         "exceptions": exceptions_list  # Optional: return the list of exceptions if needed
#     }


# def process_files(input_dir, output_dir):
#     """
#     Process files from input directory, categorize them, extract information,
#     and store results in the database according to the provided schema.
    
#     Returns:
#         dict: Result containing status code, message, and statistics
#     """
#     api_key = ""
#     if not api_key:
#         logger.error("API key not found. Cannot proceed with document processing.")
#         return {
#             "status_code": 400,
#             "message": "API key not found. Cannot proceed with document processing.",
#             "total_check_amount": 0,
#             "successful_docs": 0,
#             "exception_docs": 0
#         }
    
#     # Initialize counters and tracking variables
#     successful_docs = 0
#     exception_docs = 0
#     check_amounts = []
#     exceptions_list = []
    
#     base_dir = "Files"
#     categories = ["check", "mortgage", "claim", "coupons", "agency", "no money - cancel", "unknown"]
    
#     # Ensure document type IDs are in the database
#     try:
#         category_ids_map = ensure_document_types_in_db(categories)
#     except Exception as e:
#         logger.error(f"Failed to ensure document types in database: {str(e)}")
#         return {
#             "status_code": 500,
#             "message": f"Failed to prepare document types: {str(e)}",
#             "total_check_amount": 0,
#             "successful_docs": 0,
#             "exception_docs": 0
#         }
    
#     for cat in categories:
#         for status in ["passed", "failed"]:
#             os.makedirs(os.path.join(base_dir, cat, status, "files"), exist_ok=True)
#             os.makedirs(os.path.join(base_dir, cat, status, "jsons"), exist_ok=True)
    
#     json_tif_pairs = []
    
#     # Generate a unique batch name based on timestamp
#     batch_name = f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
#     # Create a batch in the database
#     try:
#         batch_id = create_batch_in_db(batch_name)
#         logger.info(f"Created batch with ID: {batch_id}, name: {batch_name}")
#     except Exception as e:
#         logger.error(f"Failed to create batch in database: {str(e)}")
#         return {
#             "status_code": 500,
#             "message": f"Failed to create batch in database: {str(e)}",
#             "total_check_amount": 0,
#             "successful_docs": 0,
#             "exception_docs": 0
#         }
    
#     files_dir = os.path.join("Files", "RAW Data")
#     for fileObj in (input_dir):
#         filename = fileObj.filename
#         file_path = os.path.join(files_dir, filename)
#         if not filename.lower().endswith((".tif", ".tiff", ".pdf")):
#             continue
        
#         print(f"Processing {filename}...")
        
#         text = extract_text_from_tiff(fileObj) if filename.lower().endswith((".tif", ".tiff")) else extract_text_from_pdf(fileObj)
#         if not text:
#             logger.warning(f"No text extracted from {filename}")
#             exception_reason = "No text could be extracted from file"
            
#             # Create document entry with "exception" status
#             try:
#                 doc_id = insert_document_in_db(
#                     batch_id=batch_id,
#                     doc_name=filename,
#                     type_id=category_ids_map.get("unknown"),
#                     status="exception"
#                 )
                
#                 # Add exception record
#                 add_exception_in_db(
#                     document_id=doc_id,
#                     exception_message=exception_reason
#                 )
                
#                 exceptions_list.append({"filename": filename, "reason": exception_reason})
#                 exception_docs += 1
#             except Exception as e:
#                 logger.error(f"Database error for {filename}: {str(e)}")
            
#             continue
        
#         category = classify_document(text)
#         json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        
#         if category.lower() != "no money - cancel":
#             try:
#                 client = openai.OpenAI(api_key=api_key)
#                 essentials = [k for k, v in category_variations.get(category.lower(), {}).items() if v["essential"]]
#                 prompt = (
#                     f"Extract all possible meaningful key-value pairs from this {category} document as JSON. "
#                     f"Ensure these essential keys are included (empty string if not found): {', '.join(essentials)}. "
#                     "Capture all relevant details such as names, numbers, dates, addresses, amounts, etc., that fit the {category} context. "
#                     "Make sure Policy & Loan numbers are distinct, they are written with each other but are different. "
#                     "Focus on accuracy and relevance, avoiding generic or unrelated pairs.\n\n"
#                     f"Document: {text[:2000]}\n\nJSON:"
#                 )
#                 response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=[{"role": "user", "content": prompt}],
#                     response_format={"type": "json_object"},
#                     max_tokens=1500
#                 )
#                 key_values = json.loads(response.choices[0].message.content)
                
#                 variations = category_variations.get(category.lower(), {})
#                 extracted = search_key(key_values, list(variations.keys()), {k: v["variations"] for k, v in variations.items()})
#                 essentials_dict = {k: "" for k, v in variations.items() if v["essential"]}
#                 key_values = {**essentials_dict, **key_values, **extracted}
                
#                 json_data = {"filename": filename, "category": category, "Important_Info": key_values}
#                 try:
#                     with open(json_path, "w") as f:
#                         json.dump(json_data, f, indent=4, cls=DecimalEncoder)
#                     json_tif_pairs.append((json_path, file_path))
                    
#                     if category.lower() == "check":
#                         amount = extracted.get("amount", Decimal('0'))
#                         if amount:
#                             try:
#                                 formatted_amount = Decimal(amount).quantize(Decimal('0.01'))
#                                 print(f"Check amount from {filename}: ${formatted_amount}\n")
#                                 check_amounts.append(Decimal(amount))
#                             except (InvalidOperation, TypeError) as e:
#                                 logger.warning(f"Could not format check amount for {filename}: {str(e)}")
#                                 exception_reason = f"Invalid check amount format: {str(e)}"
                                
#                                 # Create document entry with appropriate status
#                                 doc_id = insert_document_in_db(
#                                     batch_id=batch_id,
#                                     doc_name=filename,
#                                     type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
#                                     status="exception"
#                                 )
                                
#                                 # Add exception record
#                                 add_exception_in_db(
#                                     document_id=doc_id,
#                                     exception_message=exception_reason
#                                 )
                                
#                                 exceptions_list.append({"filename": filename, "reason": exception_reason})
#                                 exception_docs += 1
#                 except (IOError, OSError) as e:
#                     logger.error(f"Failed to write JSON file {json_path}: {str(e)}")
#                     exception_reason = f"Failed to write JSON file: {str(e)}"
                    
#                     # Create document entry with appropriate status
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=filename,
#                         type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
#                         status="exception"
#                     )
                    
#                     # Add exception record
#                     add_exception_in_db(
#                         document_id=doc_id,
#                         exception_message=exception_reason
#                     )
                    
#                     exceptions_list.append({"filename": filename, "reason": exception_reason})
#                     exception_docs += 1
                
#             except Exception as e:
#                 logger.error(f"Failed to process {filename} for key-value extraction: {str(e)}")
#                 exception_reason = f"Key-value extraction failed: {str(e)}"
                
#                 # Create document entry with appropriate status
#                 doc_id = insert_document_in_db(
#                     batch_id=batch_id,
#                     doc_name=filename,
#                     type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
#                     status="exception"
#                 )
                
#                 # Add exception record
#                 add_exception_in_db(
#                     document_id=doc_id,
#                     exception_message=exception_reason
#                 )
                
#                 exceptions_list.append({"filename": filename, "reason": exception_reason})
#                 exception_docs += 1
#                 continue
#         else:
#             json_tif_pairs.append((None, file_path))
    
#     total_check_amount = Decimal('0')
#     if check_amounts:
#         try:
#             total_check_amount = sum(check_amounts)
#             formatted_total = total_check_amount.quantize(Decimal('0.01'))
#             print(f"Total sum of all check amounts: ${formatted_total}")
#         except Exception as e:
#             logger.error(f"Failed to calculate total check amount: {str(e)}")
    
#     for json_path, tif_path in json_tif_pairs:
#         if json_path:
#             try:
#                 with open(json_path, 'r') as f:
#                     data = json.load(f)

#                 category = data["category"].lower()
#                 cat_dir = os.path.join(base_dir, category)
#                 variations = category_variations.get(category, {})
#                 results = search_key(data["Important_Info"], list(variations.keys()), {k: v["variations"] for k, v in variations.items()}) if variations else {}
                
#                 passed = False
#                 reasons = []
                
#                 if category == "check":
#                     if not results.get("policynumber", ""):
#                         reasons.append("Missing policynumber")
#                     if not results.get("loannumber", ""):
#                         reasons.append("Missing loannumber")
#                     if results.get("amount", "0") == "0":
#                         reasons.append("Invalid amount (zero or missing)")
#                     passed = not reasons
#                 elif category == "mortgage":
#                     required_keys = ["policynumber", "name", "address"]
#                     for key in required_keys:
#                         if not results.get(key, ""):
#                             reasons.append(f"Missing {key}")
#                     passed = not reasons
#                 elif category == "claim":
#                     required_keys = ["claimnumber", "name", "address"]
#                     for key in required_keys:
#                         if not results.get(key, ""):
#                             reasons.append(f"Missing {key}")
#                     passed = not reasons
#                 elif category == "coupons":
#                     if not results.get("policynumber", ""):
#                         reasons.append("Missing policynumber")
#                     if not results.get("insured", ""):
#                         reasons.append("Missing insured")
#                     if results.get("amount", "0") == "0":
#                         reasons.append("Invalid amount (zero or missing)")
#                     passed = not reasons
#                 elif category == "agency":
#                     if not (results.get("agencyname", "") or results.get("producer", "")):
#                         reasons.append("Missing both agencyname and producer")
#                     passed = not reasons
                
#                 # Insert document into database
#                 try:
#                     status = "processed" if passed else "exception"
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=os.path.basename(tif_path),
#                         type_id=category_ids_map.get(category, category_ids_map.get("unknown")),
#                         status=status
#                     )
                    
#                     # NEW CODE: Insert document-specific information into the appropriate table
#                     if passed:
#                         # Insert document details into the appropriate type-specific table
#                         try:
#                             if category == "check":
#                                 insert_check_document(
#                                     document_id=doc_id,
#                                     policynumber=results.get("policynumber", ""),
#                                     loannumber=results.get("loannumber", ""),
#                                     amount=Decimal(results.get("amount", "0"))
#                                 )
#                             elif category == "mortgage":
#                                 insert_mortgage_document(
#                                     document_id=doc_id,
#                                     policynumber=results.get("policynumber", ""),
#                                     name=results.get("name", ""),
#                                     address=results.get("address", "")
#                                 )
#                             elif category == "claim":
#                                 insert_claim_document(
#                                     document_id=doc_id,
#                                     claimnumber=results.get("claimnumber", ""),
#                                     name=results.get("name", ""),
#                                     address=results.get("address", "")
#                                 )
#                             elif category == "coupons":
#                                 insert_coupon_document(
#                                     document_id=doc_id,
#                                     policynumber=results.get("policynumber", ""),
#                                     insured=results.get("insured", ""),
#                                     amount=Decimal(results.get("amount", "0"))
#                                 )
#                             elif category == "agency":
#                                 insert_agency_document(
#                                     document_id=doc_id,
#                                     agencyname=results.get("agencyname", ""),
#                                     producer=results.get("producer", "")
#                                 )
#                             logger.info(f"Successfully inserted document details for {os.path.basename(tif_path)} into {category} table")
#                         except Exception as e:
#                             logger.error(f"Failed to insert details into {category} table for document {doc_id}: {str(e)}")
#                             # Add an exception for this specific error
#                             add_exception_in_db(
#                                 document_id=doc_id,
#                                 exception_message=f"Failed to insert details into {category} table: {str(e)}"
#                             )
                    
#                     # If there are failure reasons, add them as exceptions
#                     if not passed:
#                         exception_reason = ", ".join(reasons)
#                         add_exception_in_db(
#                             document_id=doc_id,
#                             exception_message=exception_reason
#                         )
#                         exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                         exception_docs += 1
#                         print(f"File {os.path.basename(tif_path)} failed for category '{category}' due to: {exception_reason}")
#                     else:
#                         successful_docs += 1
                        
#                 except Exception as e:
#                     logger.error(f"Database error for {os.path.basename(tif_path)}: {str(e)}")
#                     exception_reason = f"Database error: {str(e)}"
                    
#                     # Try to create a document entry with exception status
#                     try:
#                         doc_id = insert_document_in_db(
#                             batch_id=batch_id,
#                             doc_name=os.path.basename(tif_path),
#                             type_id=category_ids_map.get(category, category_ids_map.get("unknown")),
#                             status="exception"
#                         )
                        
#                         add_exception_in_db(
#                             document_id=doc_id,
#                             exception_message=exception_reason
#                         )
#                     except Exception as inner_e:
#                         logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                    
#                     exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                     exception_docs += 1
                
#                 dest_json_dir = os.path.join(cat_dir, "passed" if passed else "failed", "jsons")
#                 dest_file_dir = os.path.join(cat_dir, "passed" if passed else "failed", "files")
                
#                 json_copied = safe_copy_file(json_path, os.path.join(dest_json_dir, os.path.basename(json_path)))
#                 file_copied = safe_copy_file_obj(fileObj, os.path.join(dest_file_dir, os.path.basename(tif_path)))
                
#                 if not json_copied or not file_copied:
#                     logger.warning(f"Failed to move some files for {os.path.basename(json_path)}")
                    
#             except Exception as e:
#                 logger.error(f"Failed to process {os.path.basename(json_path)}: {str(e)}")
#                 cat_dir = os.path.join(base_dir, "unknown")
#                 safe_copy_file(json_path, os.path.join(cat_dir, "failed", "jsons", os.path.basename(json_path)))
#                 safe_copy_file(tif_path, os.path.join(cat_dir, "failed", "files", os.path.basename(tif_path)))
#                 print(f"File {os.path.basename(tif_path)} moved to 'unknown/failed' due to processing error: {str(e)}")
                
#                 exception_reason = f"Processing error: {str(e)}"
                
#                 # Create document entry with "exception" status
#                 try:
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=os.path.basename(tif_path),
#                         type_id=category_ids_map.get("unknown"),
#                         status="exception"
#                     )
                    
#                     add_exception_in_db(
#                         document_id=doc_id,
#                         exception_message=exception_reason
#                     )
#                 except Exception as inner_e:
#                     logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                
#                 exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                 exception_docs += 1
#         else:
#             # No Money - Cancel category case
#             dest_file_dir = os.path.join(base_dir, "no money - cancel", "passed", "files")
#             os.makedirs(dest_file_dir, exist_ok=True)
#             if not safe_copy_file(tif_path, os.path.join(dest_file_dir, os.path.basename(tif_path))):
#                 logger.error(f"Failed to move {os.path.basename(tif_path)} to no money - cancel directory")
#                 exception_reason = "Failed to move file to no money - cancel directory"
                
#                 # Create document entry with "exception" status
#                 try:
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=os.path.basename(tif_path),
#                         type_id=category_ids_map.get("no money - cancel"),
#                         status="exception"
#                     )
                    
#                     add_exception_in_db(
#                         document_id=doc_id,
#                         exception_message=exception_reason
#                     )
#                 except Exception as inner_e:
#                     logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                
#                 exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                 exception_docs += 1
#             else:
#                 print(f"File {os.path.basename(tif_path)} moved to 'no money - cancel/passed' as it was classified as 'No Money - Cancel'")
                
#                 # Insert into database as a no-money document
#                 try:
#                     doc_id = insert_document_in_db(
#                         batch_id=batch_id,
#                         doc_name=os.path.basename(tif_path),
#                         type_id=category_ids_map.get("no money - cancel"),
#                         status="processed"
#                     )
#                     successful_docs += 1
#                 except Exception as e:
#                     logger.error(f"Database error for {os.path.basename(tif_path)}: {str(e)}")
#                     exception_reason = f"Database error: {str(e)}"
                    
#                     # Try to create a document entry with exception status
#                     try:
#                         doc_id = insert_document_in_db(
#                             batch_id=batch_id,
#                             doc_name=os.path.basename(tif_path),
#                             type_id=category_ids_map.get("no money - cancel"),
#                             status="exception"
#                         )
                        
#                         add_exception_in_db(
#                             document_id=doc_id,
#                             exception_message=exception_reason
#                         )
#                     except Exception as inner_e:
#                         logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                    
#                     exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
#                     exception_docs += 1
        
#         print("\n")
    
#     logger.info("Processing complete.")
    
#     # Return the requested information
#     return {
#         "status_code": 200,
#         "message": "Document processing completed successfully",
#         "total_check_amount": str(total_check_amount.quantize(Decimal('0.01'))),
#         "successful_docs": successful_docs,
#         "exception_docs": exception_docs,
#         "exceptions": exceptions_list  # Optional: return the list of exceptions if needed
#     }

def process_files(input_dir, output_dir):
    """
    Process files from input directory, categorize them, extract information,
    and store results in the database according to the provided schema.
    
    Returns:
        dict: Result containing status code, message, and statistics
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.error("API key not found. Cannot proceed with document processing.")
        return {
            "status_code": 400,
            "message": "API key not found. Cannot proceed with document processing.",
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
    
    # Ensure document type IDs are in the database
    try:
        category_ids_map = ensure_document_types_in_db(categories)
    except Exception as e:
        logger.error(f"Failed to ensure document types in database: {str(e)}")
        return {
            "status_code": 500,
            "message": f"Failed to prepare document types: {str(e)}",
            "total_check_amount": 0,
            "successful_docs": 0,
            "exception_docs": 0
        }
    
    for cat in categories:
        for status in ["passed", "failed"]:
            os.makedirs(os.path.join(base_dir, cat, status, "files"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, cat, status, "jsons"), exist_ok=True)
    
    json_tif_pairs = []
    
    # Generate a unique batch name based on timestamp
    batch_name = f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create a batch in the database
    try:
        batch_id = create_batch_in_db(batch_name)
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
    
    files_dir = os.path.join("Files", "RAW Data")
    for fileObj in (input_dir):
        filename = fileObj.filename
        file_path = os.path.join(files_dir, filename)
        if not filename.lower().endswith((".tif", ".tiff", ".pdf")):
            continue
        
        print(f"Processing {filename}...")
        
        text = extract_text_from_tiff(fileObj) if filename.lower().endswith((".tif", ".tiff")) else extract_text_from_pdf(fileObj)
        if not text:
            logger.warning(f"No text extracted from {filename}")
            exception_reason = "No text could be extracted from file"
            
            # Create document entry with "exception" status and empty JSON
            try:
                doc_id = insert_document_in_db(
                    batch_id=batch_id,
                    doc_name=filename,
                    type_id=category_ids_map.get("unknown"),
                    status="exception",
                    json_content=json.dumps({"error": exception_reason})
                )
                
                # Add exception record
                add_exception_in_db(
                    document_id=doc_id,
                    exception_message=exception_reason
                )
                
                exceptions_list.append({"filename": filename, "reason": exception_reason})
                exception_docs += 1
            except Exception as e:
                logger.error(f"Database error for {filename}: {str(e)}")
            
            continue
        
        category = classify_document(text)
        json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        
        if category.lower() != "no money - cancel":
            try:
                client = openai.OpenAI(api_key=api_key)
                essentials = [k for k, v in category_variations.get(category.lower(), {}).items() if v["essential"]]
                prompt = (
                    f"Extract all possible meaningful key-value pairs from this {category} document as JSON. "
                    f"Ensure these essential keys are included (empty string if not found): {', '.join(essentials)}. "
                    "Capture all relevant details such as names, numbers, dates, addresses, amounts, etc., that fit the {category} context. "
                    "Make sure Policy & Loan numbers are distinct, they are written with each other but are different. "
                    "Focus on accuracy and relevance, avoiding generic or unrelated pairs.\n\n"
                    f"Document: {text[:2000]}\n\nJSON:"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=1500
                )
                key_values = json.loads(response.choices[0].message.content)
                
                variations = category_variations.get(category.lower(), {})
                extracted = search_key(key_values, list(variations.keys()), {k: v["variations"] for k, v in variations.items()})
                essentials_dict = {k: "" for k, v in variations.items() if v["essential"]}
                key_values = {**essentials_dict, **key_values, **extracted}
                
                json_data = {"filename": filename, "category": category, "Important_Info": key_values}
                json_str = json.dumps(json_data, indent=4, cls=DecimalEncoder)
                
                try:
                    with open(json_path, "w") as f:
                        f.write(json_str)
                    json_tif_pairs.append((json_path, file_path, json_str))
                    
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
                                doc_id = insert_document_in_db(
                                    batch_id=batch_id,
                                    doc_name=filename,
                                    type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
                                    status="exception",
                                    json_content=json_str
                                )
                                
                                # Add exception record
                                add_exception_in_db(
                                    document_id=doc_id,
                                    exception_message=exception_reason
                                )
                                
                                exceptions_list.append({"filename": filename, "reason": exception_reason})
                                exception_docs += 1
                except (IOError, OSError) as e:
                    logger.error(f"Failed to write JSON file {json_path}: {str(e)}")
                    exception_reason = f"Failed to write JSON file: {str(e)}"
                    
                    # Create document entry with appropriate status and JSON
                    doc_id = insert_document_in_db(
                        batch_id=batch_id,
                        doc_name=filename,
                        type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
                        status="exception",
                        json_content=json_str
                    )
                    
                    # Add exception record
                    add_exception_in_db(
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
                doc_id = insert_document_in_db(
                    batch_id=batch_id,
                    doc_name=filename,
                    type_id=category_ids_map.get(category.lower(), category_ids_map.get("unknown")),
                    status="exception",
                    json_content=error_json
                )
                
                # Add exception record
                add_exception_in_db(
                    document_id=doc_id,
                    exception_message=exception_reason
                )
                
                exceptions_list.append({"filename": filename, "reason": exception_reason})
                exception_docs += 1
                continue
        else:
            # No Money - Cancel category
            json_data = {"filename": filename, "category": "no money - cancel", "Important_Info": {}}
            json_str = json.dumps(json_data, indent=4)
            json_tif_pairs.append((None, file_path, json_str))
    
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
                cat_dir = os.path.join(base_dir, category)
                variations = category_variations.get(category, {})
                results = search_key(data["Important_Info"], list(variations.keys()), {k: v["variations"] for k, v in variations.items()}) if variations else {}
                
                passed = False
                reasons = []
                
                if category == "check":
                    if not results.get("policynumber", ""):
                        reasons.append("Missing policynumber")
                    if not results.get("loannumber", ""):
                        reasons.append("Missing loannumber")
                    if results.get("amount", "0") == "0":
                        reasons.append("Invalid amount (zero or missing)")
                    passed = not reasons
                elif category == "mortgage":
                    required_keys = ["policynumber", "name", "address"]
                    for key in required_keys:
                        if not results.get(key, ""):
                            reasons.append(f"Missing {key}")
                    passed = not reasons
                elif category == "claim":
                    required_keys = ["claimnumber", "name", "address"]
                    for key in required_keys:
                        if not results.get(key, ""):
                            reasons.append(f"Missing {key}")
                    passed = not reasons
                elif category == "coupons":
                    if not results.get("policynumber", ""):
                        reasons.append("Missing policynumber")
                    if not results.get("insured", ""):
                        reasons.append("Missing insured")
                    if results.get("amount", "0") == "0":
                        reasons.append("Invalid amount (zero or missing)")
                    passed = not reasons
                elif category == "agency":
                    if not (results.get("agencyname", "") or results.get("producer", "")):
                        reasons.append("Missing both agencyname and producer")
                    passed = not reasons
                
                # Insert document into database
                try:
                    status = "processed" if passed else "exception"
                    doc_id = insert_document_in_db(
                        batch_id=batch_id,
                        doc_name=os.path.basename(tif_path),
                        type_id=category_ids_map.get(category, category_ids_map.get("unknown")),
                        status=status,
                        json_content=json_str
                    )
                    
                    # Insert document-specific information into the appropriate table
                    if passed:
                        # Insert document details into the appropriate type-specific table
                        try:
                            if category == "check":
                                insert_check_document(
                                    document_id=doc_id,
                                    policynumber=results.get("policynumber", ""),
                                    loannumber=results.get("loannumber", ""),
                                    amount=Decimal(results.get("amount", "0"))
                                )
                            elif category == "mortgage":
                                insert_mortgage_document(
                                    document_id=doc_id,
                                    policynumber=results.get("policynumber", ""),
                                    name=results.get("name", ""),
                                    address=results.get("address", "")
                                )
                            elif category == "claim":
                                insert_claim_document(
                                    document_id=doc_id,
                                    claimnumber=results.get("claimnumber", ""),
                                    name=results.get("name", ""),
                                    address=results.get("address", "")
                                )
                            elif category == "coupons":
                                insert_coupon_document(
                                    document_id=doc_id,
                                    policynumber=results.get("policynumber", ""),
                                    insured=results.get("insured", ""),
                                    amount=Decimal(results.get("amount", "0"))
                                )
                            elif category == "agency":
                                insert_agency_document(
                                    document_id=doc_id,
                                    agencyname=results.get("agencyname", ""),
                                    producer=results.get("producer", "")
                                )
                            logger.info(f"Successfully inserted document details for {os.path.basename(tif_path)} into {category} table")
                        except Exception as e:
                            logger.error(f"Failed to insert details into {category} table for document {doc_id}: {str(e)}")
                            # Add an exception for this specific error
                            add_exception_in_db(
                                document_id=doc_id,
                                exception_message=f"Failed to insert details into {category} table: {str(e)}"
                            )
                    
                    # If there are failure reasons, add them as exceptions
                    if not passed:
                        exception_reason = ", ".join(reasons)
                        add_exception_in_db(
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
                    
                    # Try to create a document entry with exception status
                    try:
                        doc_id = insert_document_in_db(
                            batch_id=batch_id,
                            doc_name=os.path.basename(tif_path),
                            type_id=category_ids_map.get(category, category_ids_map.get("unknown")),
                            status="exception",
                            json_content=json_str
                        )
                        
                        add_exception_in_db(
                            document_id=doc_id,
                            exception_message=exception_reason
                        )
                    except Exception as inner_e:
                        logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                    
                    exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
                    exception_docs += 1
                
                dest_json_dir = os.path.join(cat_dir, "passed" if passed else "failed", "jsons")
                dest_file_dir = os.path.join(cat_dir, "passed" if passed else "failed", "files")
                
                json_copied = safe_copy_file(json_path, os.path.join(dest_json_dir, os.path.basename(json_path)))
                file_copied = safe_copy_file_obj(fileObj, os.path.join(dest_file_dir, os.path.basename(tif_path)))
                
                if not json_copied or not file_copied:
                    logger.warning(f"Failed to move some files for {os.path.basename(json_path)}")
                    
            except Exception as e:
                logger.error(f"Failed to process {os.path.basename(json_path)}: {str(e)}")
                cat_dir = os.path.join(base_dir, "unknown")
                safe_copy_file(json_path, os.path.join(cat_dir, "failed", "jsons", os.path.basename(json_path)))
                safe_copy_file(tif_path, os.path.join(cat_dir, "failed", "files", os.path.basename(tif_path)))
                print(f"File {os.path.basename(tif_path)} moved to 'unknown/failed' due to processing error: {str(e)}")
                
                exception_reason = f"Processing error: {str(e)}"
                
                # Create document entry with "exception" status and error JSON
                error_json = json.dumps({"error": exception_reason, "filename": os.path.basename(tif_path)})
                try:
                    doc_id = insert_document_in_db(
                        batch_id=batch_id,
                        doc_name=os.path.basename(tif_path),
                        type_id=category_ids_map.get("unknown"),
                        status="exception",
                        json_content=error_json
                    )
                    
                    add_exception_in_db(
                        document_id=doc_id,
                        exception_message=exception_reason
                    )
                except Exception as inner_e:
                    logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                
                exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
                exception_docs += 1
        else:
            # No Money - Cancel category case
            dest_file_dir = os.path.join(base_dir, "no money - cancel", "passed", "files")
            os.makedirs(dest_file_dir, exist_ok=True)
            if not safe_copy_file(tif_path, os.path.join(dest_file_dir, os.path.basename(tif_path))):
                logger.error(f"Failed to move {os.path.basename(tif_path)} to no money - cancel directory")
                exception_reason = "Failed to move file to no money - cancel directory"
                
                # Create document entry with "exception" status and error JSON
                error_json = json.dumps({"error": exception_reason, "filename": os.path.basename(tif_path), "category": "no money - cancel"})
                try:
                    doc_id = insert_document_in_db(
                        batch_id=batch_id,
                        doc_name=os.path.basename(tif_path),
                        type_id=category_ids_map.get("no money - cancel"),
                        status="exception",
                        json_content=error_json
                    )
                    
                    add_exception_in_db(
                        document_id=doc_id,
                        exception_message=exception_reason
                    )
                except Exception as inner_e:
                    logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                
                exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
                exception_docs += 1
            else:
                print(f"File {os.path.basename(tif_path)} moved to 'no money - cancel/passed' as it was classified as 'No Money - Cancel'")
                
                # Insert into database as a no-money document
                try:
                    doc_id = insert_document_in_db(
                        batch_id=batch_id,
                        doc_name=os.path.basename(tif_path),
                        type_id=category_ids_map.get("no money - cancel"),
                        status="processed",
                        json_content=json_str
                    )
                    successful_docs += 1
                except Exception as e:
                    logger.error(f"Database error for {os.path.basename(tif_path)}: {str(e)}")
                    exception_reason = f"Database error: {str(e)}"
                    
                    # Try to create a document entry with exception status
                    error_json = json.dumps({"error": exception_reason, "filename": os.path.basename(tif_path), "category": "no money - cancel"})
                    try:
                        doc_id = insert_document_in_db(
                            batch_id=batch_id,
                            doc_name=os.path.basename(tif_path),
                            type_id=category_ids_map.get("no money - cancel"),
                            status="exception",
                            json_content=error_json
                        )
                        
                        add_exception_in_db(
                            document_id=doc_id,
                            exception_message=exception_reason
                        )
                    except Exception as inner_e:
                        logger.error(f"Failed to record exception in DB: {str(inner_e)}")
                    
                    exceptions_list.append({"filename": os.path.basename(tif_path), "reason": exception_reason})
                    exception_docs += 1
        
        print("\n")
    
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
    print("Hi")
    try:
        print("Hi")
        success = process_files(tiff_dir, output_dir)
        print("Hi")
        if success:
            print("Processing completed successfully.")
        else:
            print("Processing finished with errors. Check log for details.")
    except Exception as e:
        logger.critical(f"Fatal error during processing: {str(e)}")
        print("Processing failed with critical error. See log for details.")