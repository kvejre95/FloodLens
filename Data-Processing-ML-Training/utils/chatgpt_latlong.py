'''
Code to test the LLM integration modules
'''

import requests
from openai import OpenAI
client = OpenAI()

# export OPENAI_API_KEY='your_key'

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_location",
            "description": "Get the location from the user input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"]
            },
        }
    },
]

def get_location_details(location_query):
    # Use the OpenAI API to process the natural language input
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": location_query}
        ],
        tools=tools
    )
    location = response.choices[0].message.tool_calls[0].function.arguments
    return location.strip()

def geocode_location(location_name):
    # Use Nominatim service to convert the location name into coordinates
    headers = {
        'User-Agent': 'FloodLense'
    }
    base_url = 'https://nominatim.openstreetmap.org/search'
    params = {'q': location_name, 'format': 'json'}
    response = requests.get(base_url, headers=headers, params=params)
    data = response.json()
    # Extract the latitude and longitude from the response
    if data:
        coordinates = data[0]
        return coordinates['lat'], coordinates['lon']
    else:
        return None, None

def main():
    # Take user input
    location_query = "Hey ChatGPT, can you show the flood affected areas at Mumbai, India."
    # Get the location details
    location_name = get_location_details(location_query)
    # Geocode the location
    latitude, longitude = geocode_location(location_name)
    print('here')
    print(location_name)
    print(geocode_location(location_name))
    # Return the coordinates
    if latitude and longitude:
        print(f"The coordinates for {location_name} are: Latitude {latitude}, Longitude {longitude}")
    else:
        print("Could not find the coordinates for the location.")

if __name__ == "__main__":
    main()


