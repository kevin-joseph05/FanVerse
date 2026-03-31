# FanVerse
Install dependencies with `uv sync`. 


Run the dashboard locally by doing:

```
uv run streamlit run dashboard/app.py 
```
For the insights panel, you will need to configure your own environment variables. Do so by making a .env file in the repository folder, and adding your Google Gemini API key to the GEMINI_API_KEY field like so:

```
GEMINI_API_KEY=put-your-api-key-here
```
