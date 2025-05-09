#!/usr/bin/env python3
import os
import requests
import json
import subprocess

print("===== Railway Connection Test =====")

# First check if Railway CLI is working
print("\n1. Testing Railway CLI...")
try:
    result = subprocess.run(['railway', 'status'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Railway CLI is working correctly")
        print(result.stdout)
    else:
        print("❌ Railway CLI error:")
        print(result.stderr)
except Exception as e:
    print(f"❌ Error running Railway CLI: {str(e)}")

# Test Railway API connection
print("\n2. Testing Railway API connection...")
try:
    # Get the project ID from the railway.json file
    with open('railway.json', 'r') as f:
        railway_config = json.load(f)
        print(f"Read railway.json: {railway_config}")
    
    # Get the token from environment or prompt for it
    token = os.environ.get('RAILWAY_TOKEN')
    if not token:
        token = input("Enter your Railway API token: ")
        os.environ['RAILWAY_TOKEN'] = token
    
    # Make a test API call
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Use GraphQL to get project info
    query = '''
    query {
      projects {
        edges {
          node {
            id
            name
            services {
              edges {
                node {
                  id
                  name
                }
              }
            }
          }
        }
      }
    }
    '''
    
    response = requests.post(
        'https://backboard.railway.app/graphql/v2',
        headers=headers,
        json={'query': query}
    )
    
    if response.status_code == 200:
        data = response.json()
        if 'errors' in data:
            print(f"❌ API Error: {json.dumps(data['errors'], indent=2)}")
        else:
            print("✅ Railway API connection successful!")
            projects = data['data']['projects']['edges']
            print(f"Found {len(projects)} projects:")
            for project in projects:
                print(f"  - {project['node']['name']} (ID: {project['node']['id']})")
                services = project['node']['services']['edges']
                for service in services:
                    print(f"    Service: {service['node']['name']} (ID: {service['node']['id']})")
    else:
        print(f"❌ API Request Failed: Status {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Exception: {str(e)}")

print("\n3. Testing MCP Configuration...")
try:
    # Check if MCP is already installed
    result = subprocess.run(['npx', '@smithery/cli@latest', 'list'], capture_output=True, text=True)
    if 'railway-mcp' in result.stdout:
        print("✅ Railway MCP is installed")
        print(result.stdout)
    else:
        print("❓ Railway MCP might not be installed correctly")
        print(result.stdout)
except Exception as e:
    print(f"❌ Error checking MCP installation: {str(e)}")

print("\n===== Test Complete =====") 