import fitz  # PyMuPDF
import qdrant_client
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import base64
import uuid

# Initialize Qdrant client and SentenceTransformer
qdrant_client = qdrant_client.QdrantClient(url="http://localhost:6333")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the vector size based on the SentenceTransformer model
vector_size = model.get_sentence_embedding_dimension()

# Create a collection in Qdrant with vectors_config
collection_name = "cp_profile"
try:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.DOT
        )
    )
except Exception as e:
    print(f"Collection creation failed or already exists: {e}")

def save_pdf_to_qdrant(pdf_path):
    doc = fitz.open(pdf_path)
    
    # Process each page in the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        print(f"Processing page {page_num + 1} of {pdf_path}")
        
        # Save text with unique ID
        text_embedding = model.encode(text).tolist()
        text_id = str(uuid.uuid4())  # Generate a unique ID for the text point
        
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(id=text_id, vector=text_embedding, payload={"page_text": text})
                ]
            )
        except Exception as e:
            print(f"Failed to upsert text data: {e}")
        
        '''
        # Save images (currently commented out)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_data = base64.b64encode(image_bytes).decode('utf-8')
            
            # Image embedding (you might want to use a model for image embeddings instead of text transformer)
            image_embedding = model.encode(image_data).tolist()
            image_id = str(uuid.uuid4())  # Generate a unique ID for the image point
            
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(id=image_id, vector=image_embedding, payload={"image": image_data})
                    ]
                )
            except Exception as e:
                print(f"Failed to upsert image data: {e}")
        '''
    
    print(f"PDF '{pdf_path}' has been successfully processed and stored in Qdrant.")

# Example usage for processing multiple PDFs
save_pdf_to_qdrant("D:\ClouxiPlexi\CP_chatbot\cp_profile1.pdf")
save_pdf_to_qdrant("D:\ClouxiPlexi\CP_chatbot\cp_profile2.pdf")
save_pdf_to_qdrant("D:\ClouxiPlexi\CP_chatbot\cp_profile3.pdf")
