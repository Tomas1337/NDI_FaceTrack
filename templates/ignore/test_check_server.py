import sys
import requests
from fastapi.testclient import TestClient
import uvicorn
import pytest


from config import CONFIG
from multiprocessing import Process
from TrackingGUI import check_server
from TrackingServer_FastAPI import app



def run_server():
    uvicorn.run(app)

@pytest.fixture
def server():
    proc = Process(target=run_server, args=(), daemon=True)
    proc.start() 
    yield
    proc.kill() # Cleanup after test

def test_check_server(server):
    response = check_server()
    assert (response == True)

if __name__ == '__main__':
    test_check_server(server)
