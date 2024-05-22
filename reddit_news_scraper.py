from flask import Flask, jsonify, request, redirect, render_template
import requests
from apify_client import ApifyClient
import os
import json




app = Flask(__name__)

APIFY_TOKEN = 'apify_api_XDR6FUjnm84BGgjaPgdiJTqW0nWUrQ0uYXID'
REDDIT_SCRAPER_ACTOR_ID = 'oAuCIx3ItNrs2okjQ'

# Initialize Apify client
apify_client = ApifyClient(token=APIFY_TOKEN)

@app.route('/index_data_scraper')
def index_data_scraper():
    return '<form action="/scrape_reddit" method="post"><input type="text" name="url"><input type="submit" value="Submit"></form>'


@app.route('/scrape_reddit', methods=['POST'])
def scrape_reddit(url):
    
    # Prepare the Actor input
    run_input = {
        "startUrls": [{"url": url}],
        "maxItems": 10000,
        "maxPostCount": 100,
        "maxComments": 100,
        "maxCommunitiesCount": 0,
        "maxUserCount": 0,
        "scrollTimeout": 40,
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
        },
    }

    # Call the Reddit Scraper Lite actor
    response = apify_client.actor(REDDIT_SCRAPER_ACTOR_ID).call(run_input=run_input)

    dataset_id = response.get('defaultDatasetId')

    api_url = f'https://api.apify.com/v2/datasets/{dataset_id}/items'
    data_response = requests.get(api_url)

        # Check if request is successful
    if data_response.status_code == 200:
        # Save data to a JSON file
        data = data_response.json()
        file_path = os.path.join(app.root_path, 'scraped_data.json')
        with open(file_path, 'w') as f:
            json.dump(data, f)

        return render_template('redirect.html')
    else:
        return 'Failed to fetch data from Apify API', 500




if __name__ == '__main__':
    app.run(debug=True)

