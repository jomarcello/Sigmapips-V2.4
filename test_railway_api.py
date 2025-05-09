import os
import requests
import json

token = os.environ.get('RAILWAY_TOKEN')
project_id = os.environ.get('RAILWAY_PROJECT_ID')

query = '''
query {
  project(id: "%s") {
    name
    services {
      edges {
        node {
          name
        }
      }
    }
  }
}
''' % project_id

headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

try:
    response = requests.post(
        'https://backboard.railway.app/graphql/v2',
        headers=headers,
        json={'query': query}
    )
    
    if response.status_code == 200:
        data = response.json()
        if 'errors' in data:
            print(f'❌ API Fout: {json.dumps(data["errors"])}')
        else:
            print(f'✅ Succesvol verbonden met project: {data["data"]["project"]["name"]}')
            services = data['data']['project']['services']['edges']
            print(f'Services in dit project:')
            for service in services:
                print(f'  - {service["node"]["name"]}')
    else:
        print(f'❌ API Request Gefaald: Status {response.status_code}')
        print(response.text)
except Exception as e:
    print(f'❌ Exception: {str(e)}')
