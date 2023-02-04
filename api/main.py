import os
import requests
from fastapi import FastAPI
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse


TF_SERVER = os.environ.get("TF_SERVER", "http://localhost:8501")

app = FastAPI(
    title="Recsys",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
def docs_redirect():
    return "/docs"


@app.get(path="/v1/recommendation/model/two_tower", )
def two_tower(customer_id: int = Query(default=1658781, example=1658781)):
    r = requests.post(
        url=f'{TF_SERVER}/v1/models/two_tower:predict',
        json={"instances": [{"CUSTOMER_ID": customer_id}]}
    )
    data = r.json()
    data = [
        {"score": s, "item": i}
        for s, i in zip(data["predictions"][0]["scores"], data["predictions"][0]["items"])
    ]
    return data


@app.get(path="/v1/recommendation/model/two_tower_context")
def two_tower_context(
        customer_id: int = Query(example=1658781),
        age_group: str = Query(example="30-34")
    ):
    r = requests.post(
        url=f'{TF_SERVER}/v1/models/two_tower_context:predict',
        json={
            "instances": [
                {
                    "CUSTOMER_ID": customer_id,
                    "AGE_GROUP": age_group,
                }
            ]
        }
    )
    data = r.json()
    data = [
        {"score": s, "item": i}
        for s, i in zip(data["predictions"][0]["scores"], data["predictions"][0]["items"])
    ]
    return data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug", reload=False)
