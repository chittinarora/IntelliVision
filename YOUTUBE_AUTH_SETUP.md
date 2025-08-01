# YouTube Authentication Setup

This guide helps you set up YouTube authentication to bypass bot detection when processing videos.

## Quick Fix - Anti-Detection Improvements

The system now includes:
- Browser user-agent spoofing
- Proper HTTP headers mimicking real browsers
- Skip HLS/DASH formats that trigger bot detection
- Optional cookie support for authenticated requests

## For Videos Requiring Authentication

Some YouTube videos require authentication. If you encounter errors like:
```
Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies
```

Follow these steps:

### Option 1: Browser Cookies (Recommended)

1. **Export cookies from your browser:**
   - Install browser extension "Get cookies.txt" for Chrome/Firefox
   - Login to YouTube in your browser
   - Visit the problematic video page
   - Use the extension to export cookies as `cookies.txt`

2. **Place cookies file:**
   ```bash
   # Copy cookies.txt to your Docker container
   docker cp cookies.txt intellivision-web-1:/app/intellivision/cookies.txt
   
   # Or set custom path via environment variable
   YOUTUBE_COOKIES_PATH=/path/to/your/cookies.txt
   ```

3. **Restart services:**
   ```bash
   docker-compose restart web celery-worker
   ```

### Option 2: Manual Cookie Export

If browser extensions don't work:

1. **Get cookies manually:**
   - Login to YouTube
   - Open Developer Tools (F12)
   - Go to Application/Storage → Cookies → https://www.youtube.com
   - Copy relevant cookies (session, consent, etc.)

2. **Create cookies.txt file:**
   ```
   # Netscape HTTP Cookie File
   .youtube.com	TRUE	/	FALSE	0	cookie_name	cookie_value
   ```

## Testing

After setup, test with:
```bash
# Check if cookies are detected
docker-compose logs web | grep "authentication cookies"

# Should show: "Using authentication cookies for YouTube download."
```

## Troubleshooting

- **Still getting bot detection?** Try different browser/IP or wait a few hours
- **Cookies not working?** Ensure cookies.txt is in Netscape format
- **File not found?** Check Docker volume mounts and file permissions

## Environment Variables

```env
# Optional: Custom cookies path
YOUTUBE_COOKIES_PATH=/app/intellivision/cookies.txt
```

The system will automatically detect and use cookies if available at the specified path.