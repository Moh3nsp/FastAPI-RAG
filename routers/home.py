from fastapi import APIRouter, Depends, status, HTTPException, File, UploadFile
import shutil
from models.BaseUploadForm import BaseUploadForm
from rag_infrastructure.tools import get_answer, get_response, save_vectors
import os

UPLOAD_DIRECTORY='doc_files'

router = APIRouter(
    prefix="/Home",
    tags=['Home']
)

@router.post("/save")
async def save(from_data:  BaseUploadForm = Depends()):
    
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)    
    
    for file in from_data.files:        
            
        try:
            file_location = f"{UPLOAD_DIRECTORY}/{ file.filename}"

            with open(file_location, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)

            vector_resp = save_vectors(file_location)    

        except Exception as e:
            return {"message": f"There was an error uploading the file: {str(e)}"}

        finally:
            await file.close()  

    return {"message": f"Successfully uploaded {len(from_data.files)} files", "vector_result": vector_resp }


@router.get('/')
def index(request:str): 
    return get_response(request)

@router.get('/ask')
def ask(query):
    answer = get_answer(query);
    return {"result": answer}
        