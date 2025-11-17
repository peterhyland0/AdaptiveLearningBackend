import os
import tempfile
from datetime import datetime
from http.client import HTTPException

import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from firebase_admin import auth
from google.cloud.firestore_v1 import ArrayUnion

# Initialize Firebase Admin SDK
cred = credentials.Certificate(
    "app/firebaseHandling/adaptive-learning-app-example.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "adaptive-learning-app-example.firebasestorage.app"
})

db = firestore.client()
bucket = storage.bucket()

import time
from firebase_admin import firestore


def extract_text_from_image(file_content):
    """
    Extract text from an image by uploading to GCS and querying Firestore.

    Args:
        file_content (bytes): Binary content of the image file

    Returns:
        str: Extracted text if found, None otherwise
    """
    tmp_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20250327_123456
        base_name = os.path.basename(tmp_path).replace('.jpg', '')  # Remove existing .jpg if present
        storage_destination = f"images/image_{base_name}_{timestamp}.jpg"

        # Upload to GCS
        blob = bucket.blob(storage_destination)
        blob.upload_from_filename(tmp_path)
        storage_url = f"gs://{bucket.name}/{storage_destination}"
        print(f"Uploaded image to Storage at: {storage_url}")

        # Query Firestore
        collection_ref = db.collection("extractedText")
        query = collection_ref.where(filter=firestore.FieldFilter("file", "==", storage_url))

        # Retry logic to wait for extraction process
        max_retries = 5
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            docs = query.stream()
            found = False
            for doc in docs:
                data = doc.to_dict()
                extracted_text = data.get("extractedText") or data.get("text")
                if extracted_text:
                    # print(f"Found text in Firestore on attempt {attempt + 1}: {extracted_text}")
                    return extracted_text
                found = True
                break  # Exit inner loop if we found a matching document

            if not found and attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: No matching document yet, waiting...")
                time.sleep(retry_delay)

        print("No extracted text found in Firestore after all retries")
        return None

    except firebase_admin.exceptions.FirebaseError as fe:
        print(f"Firebase error: {str(fe)}")
        return None
    except IOError as ioe:
        print(f"IO error handling file: {str(ioe)}")
        return None
    except Exception as e:
        print(f"Unexpected error processing image: {str(e)}")
        return None

    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                print(f"Error removing temporary file {tmp_path}: {str(e)}")


def create_user(request):
    # Normalize the email to ensure it's in proper format.
    email = request.email.strip().lower()
    print(f"Creating user with email: {email}")

    user_record = auth.create_user(
        email=email,
        password=request.password
    )

    db.collection('users').document(user_record.uid).set({
        "email": user_record.email,
        "admin": request.admin,
        "createdAt": SERVER_TIMESTAMP,
    })

    if hasattr(request, "adminUid") and request.adminUid:
        admin_ref = db.collection('users').document(request.adminUid)
        admin_ref.update({
            "my_students": ArrayUnion([user_record.uid])
        })

    return user_record


def delete_user(user_uid):
    try:
        # Delete the user from Firebase Authentication.
        auth.delete_user(user_uid)
        # Optionally delete the user document from Firestore.
        db.collection('users').document(user_uid).delete()
        print(f"User {user_uid} deleted successfully.")
        return True
    except Exception as e:
        print(f"Error deleting user {user_uid}: {e}")
        raise


def get_admin_students(admin_uid):
    admin_doc = db.collection("users").document(admin_uid).get()
    if not admin_doc.exists:
        raise HTTPException(status_code=404, detail="Admin not found")
    admin_data = admin_doc.to_dict()
    student_uids = admin_data.get("my_students", [])
    students = []
    for uid in student_uids:
        student_doc = db.collection("users").document(uid).get()
        if student_doc.exists:
            student_data = student_doc.to_dict()
            students.append({
                "uid": uid,
                "email": student_data.get("email", ""),
            })
    return students


def create_module_with_submodules(created_by, module_data, submodules_data):
    """
    Creates a module and its associated submodules in Firestore.

    Args:
        created_by (str): User ID who created the module.
        module_data (dict): Details of the module (name, description, etc.).
        submodules_data (list): List of submodules to be created.

    Returns:
        dict: Created module with references to its submodules.
    """
    try:
        # Create the module document and get its ID
        module_doc = db.collection('modules').document()
        module_data['createdBy'] = [created_by]
        module_data['submodules'] = []  # Placeholder for submodule references
        module_data['createdAt'] = SERVER_TIMESTAMP

        module_doc.set(module_data)
        module_id = module_doc.id

        print(f"Module created with ID: {module_id}")

        # Add submodules and collect their document IDs
        submodule_ids = []
        for submodule in submodules_data:
            submodule['moduleId'] = module_id  # Set the parent module ID
            submodule_doc = db.collection('submodules').document()
            submodule_doc.set(submodule)
            submodule_ids.append(submodule_doc.id)  # Collect submodule document IDs

            progress_data = {
                "completionDate": None,
                "completionPercentage": 0,
                "progressStatus": "Not Started",
                "lastUpdated": datetime.now().isoformat()  # Use ISO format for date-time
            }
            user_progress_doc = db.collection('userProgress').document(created_by).collection(
                'submoduleProgress').document(submodule_doc.id)
            user_progress_doc.set(progress_data)

            print(f"Submodule created with ID: {submodule_doc.id}")

        # Update the module document with submodule document IDs
        module_doc.update({'submodules': submodule_ids})
        print(f"Module updated with submodule references.")

        return {"moduleId": module_id, "submodules": submodule_ids}
    except Exception as e:
        print(f"Error creating module and submodules: {e}")
        raise


def add_users_to_module(module_id, user_ids, admin_uid):
    """
    Adds one or more user IDs to the 'createdBy' field of a module.

    Args:
        module_id (str): The ID of the module document.
        user_ids (list): A list of user IDs to add to the module.

    Returns:
        dict: A dictionary indicating success, the module id, and the added user ids.
    """
    try:
        user_ids.append(admin_uid)

        module_ref = db.collection("modules").document(module_id)
        # Add the new user IDs to the createdBy array
        module_ref.update({
            "createdBy": user_ids
        })
        print(f"Successfully added users {user_ids} to module {module_id}")

        user_ids.remove(admin_uid)

        # Retrieve the updated module document to get its submodules
        module_doc = module_ref.get()
        module_data = module_doc.to_dict()
        submodule_ids = module_data.get("submodules", [])

        # Define the default progress data for each submodule
        progress_data = {
            "completionDate": None,
            "completionPercentage": 0,
            "progressStatus": "Not Started",
            "lastUpdated": datetime.now().isoformat()  # ISO format for date-time
        }

        # For each new user, create a progress document for each submodule
        for user_id in user_ids:
            for submodule_id in submodule_ids:
                progress_ref = db.collection("userProgress").document(user_id) \
                    .collection("submoduleProgress").document(submodule_id)
                progress_ref.set(progress_data)
                print(f"Created progress doc for user {user_id} for submodule {submodule_id}")

        return {"success": True, "moduleId": module_id, "addedUsers": user_ids}
    except Exception as e:
        print(f"Error adding users to module {module_id}: {e}")
        raise e
