import requests

# Example: local server (replace with your actual host & port)
SERVER_URL = "http://192.168.31.116:8000/generate"
API_KEY = "in93dj39e39d39ei39ei3e3fle9de9die9"  # if your server needs it

def query_local_model(prompt: str):
    try:
        response = requests.post(
            SERVER_URL,
            headers={"x-api-key": API_KEY},  # remove this line if no API key is required
            json={"prompt": prompt}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "⚠️ No response field found in server reply.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    output = query_local_model(user_prompt)
    print("\n=== Model Response ===")
    print(output)
