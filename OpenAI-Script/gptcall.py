import requests
from openai import OpenAI
import json

client = OpenAI(api_key='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')

# export OPENAI_API_KEY='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'

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
    # print(location)
    return location.strip()


def geocode_location(location_name):
    # Use Nominatim service to convert the location name into coordinates
    headers = {
        'User-Agent': 'FloodLense'
    }
    # print(type(location_name))
    base_url = 'https://nominatim.openstreetmap.org/search'
    params = {'q': location_name, 'format': 'json'}
    response = requests.get(base_url, headers=headers, params=params)
    data = response.json()
    # Extract the latitude and longitude from the response
    # print(data)
    if data:
        coordinates = data[0]
        return coordinates['lat'], coordinates['lon']
    else:
        return None, None


def get_satellite_data(lat, long):
    # Use Nominatim service to convert the location name into coordinates
    # headers = {"Content-Type": "text"}
    base_url = 'http://localhost:5000/download_image'
    params = {'latitude': lat, 'longitude': long, 'format': 'json'}
    response = requests.get(base_url, params=params)
    # print(response)
    if response.status_code == 200:
        # Retrieve the URL from the response
        image_url = response.json()['url']
        print("And your annotated Image is right here - ", image_url)
    else:
        print("Failed to retrieve the image. Status Code:", response.status_code)


def main():
    # Take user input
    location_query = input("How can I help you?")
    # location_query = "Hey ChatGPT, can you show the flood affected areas at Izbat Bushra Hanna."
    # Get the location details
    location_name = get_location_details(location_query)
    location = json.loads(location_name)["location"]
    # print(location)

    # Geocode the location
    latitude, longitude = geocode_location(location)

    # Return the coordinates
    if latitude and longitude:
        print(f"Ofcourse! The coordinates for {location} are Latitude {latitude}, Longitude {longitude}")
    else:
        print("Could not find the coordinates for the location.")

    # Returns highlighted images
    get_satellite_data(latitude, longitude)


if __name__ == "__main__":
    main()
