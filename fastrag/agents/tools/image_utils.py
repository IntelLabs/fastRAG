import base64
from io import BytesIO

import requests


def get_base64_from_url(image_url):
    if image_url.startswith("http"):
        response = requests.get(
            image_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            },
        )
        buffered = BytesIO(response.content)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        with open(image_url, "rb") as image_file:
            img_str = base64.b64encode(image_file.read())

    return img_str
