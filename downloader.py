from pytube import YouTube
import sys
import ssl
import urllib.request

def download_video(url):
    print("in the method")
    if ("youtube.com" not in url):
        url = input("Enter YouTube URL: ")
    yt = YouTube(url,use_oauth=True,allow_oauth_cache=True,)
    filename = yt.title.replace(" ","_")
    print("Downloading YouTube File: " + yt.title)
    yt.streams.filter(resolution="720p").first().download(filename=filename + ".mp4")

ssl._create_default_https_context = ssl._create_unverified_context
url = sys.argv[1]
download_video(url)