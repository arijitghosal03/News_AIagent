from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os
import google.generativeai as genai
from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="News Agent API")

# Add CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables (you'll need to set these)
SERP_API_KEY = os.getenv("SERP_API_KEY")
print(f"SERP_API_KEY: {SERP_API_KEY}")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

class NewsRequest(BaseModel):
    date: str  # Format: YYYY-MM-DD
    topics: List[str]
    
class NewsArticle(BaseModel):
    title: str
    source: str
    snippet: str
    date: str
    url: Optional[str] = None
    
class NewsResponse(BaseModel):
    articles: List[NewsArticle]
    summary: str
    headline: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the News Agent API"}

@app.post("/fetch-news", response_model=NewsResponse)
async def fetch_news(request: NewsRequest):
    if not SERP_API_KEY:
        raise HTTPException(status_code=500, detail="SERP API key not configured")
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    try:
        # Validate date format
        print(f"Received request: {request}")
        try:
            datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Format topics for search
        topics_query = " ".join(request.topics)
        
        # Fetch news from SERP API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google",
                    "q": f"{topics_query} {request.date}",
                    "tbm": "nws",
                    "num": 15,
                    "api_key": SERP_API_KEY
                }
            )
        
        data = response.json()
        if "error" in data:
            raise HTTPException(status_code=400, detail=data["error"])
        
        if "news_results" not in data:
            return NewsResponse(
                articles=[],
                summary="No news found for the specified date and topics.",
                headline="No News Available"
            )
        
        # Process news results
        articles = []
        for item in data.get("news_results", [])[:15]:
            articles.append(
                NewsArticle(
                    title=item.get("title", ""),
                    source=item.get("source", ""),
                    snippet=item.get("snippet", ""),
                    date=item.get("date", request.date),
                    url=item.get("link", "")
                )
            )
        
        # Use Gemini to generate a summary and headline
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        articles_text = "\n\n".join([
            f"Title: {article.title}\nSource: {article.source}\nSnippet: {article.snippet}"
            for article in articles
        ])
        
        prompt = f"""
        Based on the following Indian news articles from  {request.date} about {', '.join(request.topics)}, 
        generate:
        1. A compelling main headline in the style of a vintage Indian newspaper (max 10 words)
        2. A concise summary of the key developments in India (100-150 words)
        
        News articles:
        {articles_text}
        
        Respond in JSON format:
        {{
            "headline": "Your headline here",
            "summary": "Your summary here"
        }}
        """
        
        response = model.generate_content(prompt)
        try:
            # Try to parse JSON from the response
            gemini_response = json.loads(response.text)
            headline = gemini_response.get("headline", "News of the Day")
            summary = gemini_response.get("summary", "Summary not available.")
        except (json.JSONDecodeError, AttributeError):
            # Fallback if JSON parsing fails
            headline = "News of the Day"
            summary = "Summary not available due to processing error."
        
        return NewsResponse(
            articles=articles,
            summary=summary,
            headline=headline
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)