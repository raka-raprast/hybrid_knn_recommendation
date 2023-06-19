import pandas as pd
from fastapi import FastAPI, Request
import psycopg2
from model import generate_recommendation

app = FastAPI()

# Database connection
conn = psycopg2.connect(
    host="your_host",
    port="your_port",
    database="your_database",
    user="your_user",
    password="your_password"
)


@app.post("/recommend")
async def recommend(request: Request):
    data = await request.json()
    user_id = data.get("user_id")

    if user_id is None:
        return {"error": "User ID is missing in the request body."}

    data = pd.read_sql_query("SELECT * FROM user_action_post", conn)
    final_recommendations = await generate_recommendation(user_id, data)
    return {"recommendations": final_recommendations}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
