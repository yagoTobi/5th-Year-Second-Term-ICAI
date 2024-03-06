from googleapiclient.discovery import build

api_key = 'AIzaSyDkpmuUEMaWrocDXcdCNLf6ZuEHQls3FDY'
youtube = build('youtube', 'v3', developerKey=api_key)

client_id = "61078976805-c18pkmuvid1c5a225255gu9edkft2rv2.apps.googleusercontent.com"
client_secret = "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"

VIDEO_ID = 'kJQP7kiw5Fk'

request = youtube.commentThreads().list(
    part="snippet",
    videoId=VIDEO_ID,
    maxResults=100  # Adjust as needed
)
response = request.execute()

for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
    print(comment)


request = youtube.videos().list(
    part="snippet,contentDetails,statistics",
    chart='mostPopular',
    regionCode='US',
    maxResults=50
)
response = request.execute()

print("\n========\n")
print("TOP VIDEOS POPULARES EN EEUU")
print("\n========\n")

for video in response['items']:
    # Process each video's data here
    print(video['snippet']['title'])  # Example: print video title
