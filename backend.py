from fastapi import FastAPI, UploadFile
app = FastAPI()

@app.post('/frame')
async def frame(frame: UploadFile):
    data = await frame.read()
    with open('latest.jpg', 'wb') as f:
        f.write(data)
    return {'ok': True}
