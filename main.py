from fastapi import FastAPI
from routers.config import register_routes
 
app = FastAPI()

register_routes(app)

