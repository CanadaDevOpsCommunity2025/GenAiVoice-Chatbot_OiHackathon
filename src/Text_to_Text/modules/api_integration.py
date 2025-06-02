import requests
from .config import API_KEY, SEARCH_ENGINE_ID

class APIIntegration:
    def __init__(self):
        self.api_key = API_KEY
        self.search_engine_id = SEARCH_ENGINE_ID

    def google_search(self, query, num_results=5):
        """Fetches search results using Google Custom Search API."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.search_engine_id,
            "num": num_results
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            search_results = []
            if "items" in data:
                for item in data["items"]:
                    search_results.append({
                        "title": item["title"],
                        "link": item["link"],
                        "snippet": item["snippet"]
                    })
            return search_results

        except requests.exceptions.RequestException as e:
            return f"(Error fetching search results: {e})"

    def fetch_live_data(self, user_input):
        """Fetches real-time data based on user input."""
        if "weather" in user_input.lower():
            try:
                res = requests.get("https://api.openweathermap.org/data/2.5/weather?q=New York&appid=your_api_key")
                return f"(Live Weather Data) {res.json()['weather'][0]['description']}, Temp: {res.json()['main']['temp']}°C"
            except:
                return "(Live Data Unavailable)"
        return "(No Live Data Found)" 