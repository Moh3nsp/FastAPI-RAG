from fastapi import File, Form, UploadFile
from typing import List


class BaseUploadForm:
    def __init__(
            self,
            files: List[UploadFile]=File(...),
            description:str=Form(...)
    ):
        self.files=files
        self.description= description