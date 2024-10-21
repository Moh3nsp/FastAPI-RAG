from fastapi import FastAPI
from . import home
routes = [
home.router
]

def register_routes(app:FastAPI):
    for route in routes:
        app.include_router(route)
   