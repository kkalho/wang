from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

codes = {
    "TEST123": {"uses": 10, "used": 0},
    "WELCOME": {"uses": 5, "used": 0},
    "ADMIN": {"uses": 999, "used": 0},
}

@app.post("/remove_watermark")
async def remove_watermark(
    file: UploadFile = File(...),
    code: str = Form(...)
):
    if code not in codes:
        raise HTTPException(status_code=400, detail="邀请码无效")
    if codes[code]["used"] >= codes[code]["uses"]:
        raise HTTPException(status_code=400, detail="次数已用完")
    codes[code]["used"] += 1

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="无法读取图片")

    h, w = image.shape[:2]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[int(h*0.85):h, int(w*0.75):w] = 255
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    io_buf = BytesIO(buffer.tobytes())
    io_buf.seek(0)
    return StreamingResponse(io_buf, media_type="image/jpeg")

@app.get("/check_remaining")
async def check_remaining(code: str):
    if code not in codes:
        return {"remaining": 0, "valid": False}
    remaining = codes[code]["uses"] - codes[code]["used"]
    return {"remaining": remaining, "valid": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

